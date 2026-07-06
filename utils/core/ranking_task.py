import os
import json

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.misc.logging_setup import logger
from utils.misc.query_saves import (
    _save_ranking_results,
    _save_retest_summary,
)
from utils.core.response_cleaning import clean_ranking_response
from utils.core.task_processor import (
    process_task,
    json_file_has_key,
)

from utils.paths.query import QueryPathConstructor

from utils.core.retest_averaging import average_ranking_trials


def run_ranking_task(
    config: SimpleNamespace,
    region_1: str,
    region_2: str,
    function: str,
    model: str,
    hemisphere_1: Optional[str],
    hemisphere_2: Optional[str],
    output_category: Optional[str],
) -> None:
    """
    Ranking analysis task

    Args:
        * config: Config object with necessary attributes
        * region_1: First brain region name
        * region_2: Second brain region name
        * function: Function name to query
        * model: Model name to use for querying
        * hemisphere_1: Hemisphere for first region ('left', 'right', or None)
        * hemisphere_2: Hemisphere for second region ('left', 'right', or None)
        * output_category: Folder category ('left', 'right', 'interhemispheric', or None)
    """
    analysis_type = "rankings"
    pair_label = f"{region_1}_vs_{region_2}"

    # Prepend the hemisphere to the region names for the LLM prompt and logging
    region_1_prompt = f"{hemisphere_1} {region_1}".strip() if hemisphere_1 else region_1
    region_2_prompt = f"{hemisphere_2} {region_2}".strip() if hemisphere_2 else region_2

    # Pass output_category as the hemisphere argument to route the save folder
    query = QueryPathConstructor(
        model=model, species=config.species,
        atlas_name=config.atlas_name,
        analysis_type=analysis_type, hemisphere=output_category,
        template_name=config.prompt_template_name,
    )

    def _trial_done(trial: Union[int, str]) -> bool:
        """
        Check if a single trial is complete by verifying the presence of
        results in the JSON file
        """
        # Check pair response JSON contains the function key for this trial
        path = query.construct_query_pair_path(
            region_1=region_1,
            region_2=region_2,
            trial=trial,
        )
        if not json_file_has_key(path=path, key=function):
            return False

        # Check for justification if needed
        if config.justify:
            just = (
                query
                .construct_query_pair_justification_path(
                    region_1=region_1,
                    region_2=region_2,
                    trial=trial,
                )
            )
            if not json_file_has_key(path=just, key=function):
                return False

        # Result and justification (if needed) are present, trial is done
        return True

    def skip_check() -> bool:
        """
        Check if all trials and final result are complete, and skip if so
        """
        # Check all retest trials
        for t in range(config.retest):
            if not _trial_done(t):
                return False

        # Check final averaged result
        if not _trial_done("final"):
            return False

        # All trials and final result are complete, can skip
        logger.info(
            f"Skipping {region_1_prompt} vs {region_2_prompt}/{function} ({model}) - "
            "already done"
        )
        return True

    def load_trial(
        trial: Union[int, str],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Load the ranking result (and justification if needed) for a single trial
        """
        # Load the ranking result from the JSON file for this trial
        path = query.construct_query_pair_path(
            region_1=region_1,
            region_2=region_2,
            trial=trial,
        )
        with open(path) as f:
            data = json.load(f)
        ranking = data[function][model]

        # Load justification if needed
        justification = None
        if config.justify:
            just_path = (
                query
                .construct_query_pair_justification_path(
                    region_1=region_1,
                    region_2=region_2,
                    trial=trial,
                )
            )
            if os.path.exists(just_path):
                with open(just_path) as f:
                    jdata = json.load(f)
                justification = (
                    jdata.get(function, {}).get(model)
                )

        # Return the ranking result and justification (if any)
        return {"ranking": ranking}, justification

    def process_response(
        answer: str, _response: str,
    ) -> Dict[str, int]:
        """
        Process the raw model response to extract the ranking result
        """
        return {
            "ranking": clean_ranking_response(answer),
        }

    def save_result(
        result: Dict[str, int],
        trial: Union[int, str],
        justification: Optional[str] = None,
    ) -> None:
        """
        Save the ranking result (and justification if needed) for a single trial
        """
        _save_ranking_results(
            region_1=region_1, region_2=region_2,
            hemisphere=output_category, function=function,
            model=model, config=config,
            ranking=result["ranking"],
            trial=trial,
            analysis_type=analysis_type,
            justification=justification,
        )

    def save_final(
        results: List[Dict[str, int]],
        justifications: List[str],
    ) -> None:
        """
        Average the ranking results across trials and save the final result
        """
        # Average the ranking values across trials and save final result
        rankings = [r["ranking"] for r in results]
        avg = average_ranking_trials(rankings=rankings)
        _save_ranking_results(
            region_1=region_1, region_2=region_2,
            hemisphere=output_category, function=function,
            model=model, config=config,
            ranking=avg["mode"],
            trial="final",
            analysis_type=analysis_type,
        )

        # if retesting was done, also save a summary of the retest results
        if config.retest > 1:
            func_summary = {
                "mode": avg["mode"],
                "counts": avg["counts"],
                "agreement_ratio": avg[
                    "agreement_ratio"
                ],
                "trials": rankings,
            }
            if justifications:
                func_summary["justifications"] = (
                    justifications
                )
            summary = {
                "num_trials": config.retest,
                "functions": {function: func_summary},
            }
            _save_retest_summary(
                region=pair_label, model=model,
                config=config, analysis_type=analysis_type,
                hemisphere=output_category,
                summary_data=summary,
            )

    # Run the task processor with the defined functions
    process_task(
        config,
        model=model,
        skip_check=skip_check,
        trial_complete=_trial_done,
        load_trial=load_trial,
        prompt_kwargs={
            "prompt_type": analysis_type,
            "region_name": pair_label,
            "function": function,          
            "hemisphere": output_category,
            "hemisphere_1": hemisphere_1,
            "hemisphere_2": hemisphere_2,
            "region_1": region_1_prompt,  # LLM template receives e.g. "left cuneus"
            "region_2": region_2_prompt,  # LLM template receives e.g. "right amygdala"
        },
        log_label=(
            f"{region_1_prompt} vs {region_2_prompt} - {function} ({model})"
        ),
        process_response=process_response,
        save_result=save_result,
        save_final=save_final,
    )
#import os
#import json
#
#from types import SimpleNamespace
#from typing import Any, Dict, List, Optional, Tuple, Union
#
#from utils.misc.logging_setup import logger
#from utils.misc.query_saves import (
#    _save_ranking_results,
#    _save_retest_summary,
#)
#from utils.core.response_cleaning import clean_ranking_response
#from utils.core.task_processor import (
#    process_task,
#    json_file_has_key,
#)
#
#from utils.paths.query import QueryPathConstructor
#
#from utils.core.retest_averaging import average_ranking_trials
#
#
#def run_ranking_task(
#    config: SimpleNamespace,
#    region_1: str,
#    region_2: str,
#    function: str,
#    model: str,
#    hemisphere: Optional[str],
#) -> None:
#    """
#    Ranking analysis task
#
#    Args:
#        * config: Config object with necessary attributes
#        * region_1: First brain region name
#        * region_2: Second brain region name
#        * function: Function name to query
#        * model: Model name to use for querying
#        * hemisphere: Hemisphere ('left', 'right', or None)
#    """
#    analysis_type = "rankings"
#    pair_label = f"{region_1}_vs_{region_2}"
#
#    query = QueryPathConstructor(
#        model=model, species=config.species,
#        atlas_name=config.atlas_name,
#        analysis_type=analysis_type, hemisphere=hemisphere,
#        template_name=config.prompt_template_name,
#    )
#
#    def _trial_done(trial: Union[int, str]) -> bool:
#        """
#        Check if a single trial is complete by verifying the presence of
#        results in the JSON file
#
#        Args:
#            * trial: Trial number (int) or 'final' for final averaged result
#
#        Returns:
#            * True if the trial is complete, False otherwise
#        """
#        # Check pair response JSON contains the function key for this trial
#        path = query.construct_query_pair_path(
#            region_1=region_1,
#            region_2=region_2,
#            trial=trial,
#        )
#        if not json_file_has_key(path=path, key=function):
#            return False
#
#        # Check for justification if needed
#        if config.justify:
#            just = (
#                query
#                .construct_query_pair_justification_path(
#                    region_1=region_1,
#                    region_2=region_2,
#                    trial=trial,
#                )
#            )
#            if not json_file_has_key(path=just, key=function):
#                return False
#
#        # Result and justification (if needed) are present, trial is done
#        return True
#
#    def skip_check() -> bool:
#        """
#        Check if all trials and final result are complete, and skip if so
#
#        Returns:
#            * True if all trials and final result are complete, False otherwise
#        """
#        # Check all retest trials
#        for t in range(config.retest):
#            if not _trial_done(t):
#                return False
#
#        # Check final averaged result
#        if not _trial_done("final"):
#            return False
#
#        # All trials and final result are complete, can skip
#        logger.info(
#            f"Skipping {region_1} vs {region_2}/{function} ({model}) - "
#            "already done"
#        )
#        return True
#
#    def load_trial(
#        trial: Union[int, str],
#    ) -> Tuple[Dict[str, Any], Optional[str]]:
#        """
#        Load the ranking result (and justification if needed) for a single
#        trial
#
#        Args:
#            * trial: Trial number (int) or 'final' for final averaged result
#
#        Returns:
#            * Tuple of (result dict, justification string or None)
#        """
#        # Load the ranking result from the JSON file for this trial
#        path = query.construct_query_pair_path(
#            region_1=region_1,
#            region_2=region_2,
#            trial=trial,
#        )
#        with open(path) as f:
#            data = json.load(f)
#        ranking = data[function][model]
#
#        # Load justification if needed
#        justification = None
#        if config.justify:
#            just_path = (
#                query
#                .construct_query_pair_justification_path(
#                    region_1=region_1,
#                    region_2=region_2,
#                    trial=trial,
#                )
#            )
#            if os.path.exists(just_path):
#                with open(just_path) as f:
#                    jdata = json.load(f)
#                justification = (
#                    jdata.get(function, {}).get(model)
#                )
#
#        # Return the ranking result and justification (if any)
#        return {"ranking": ranking}, justification
#
#    def process_response(
#        answer: str, _response: str,
#    ) -> Dict[str, int]:
#        """
#        Process the raw model response to extract the ranking result
#
#        Args:
#            * answer: Raw model response containing the ranking
#            * _response: Full raw response (including prompt)
#
#        Returns:
#            * Dict containing the cleaned ranking result
#        """
#        return {
#            "ranking": clean_ranking_response(answer),
#        }
#
#    def save_result(
#        result: Dict[str, int],
#        trial: Union[int, str],
#        justification: Optional[str] = None,
#    ) -> None:
#        """
#        Save the ranking result (and justification if needed) for a single
#        trial
#
#        Args:
#            * result: Dict containing the ranking result to save
#            * trial: Trial number (int) or 'final' for final averaged result
#            * justification: Justification string to save (if any)
#        """
#        _save_ranking_results(
#            region_1=region_1, region_2=region_2,
#            hemisphere=hemisphere, function=function,
#            model=model, config=config,
#            ranking=result["ranking"],
#            trial=trial,
#            analysis_type=analysis_type,
#            justification=justification,
#        )
#
#    def save_final(
#        results: List[Dict[str, int]],
#        justifications: List[str],
#    ) -> None:
#        """
#        Average the ranking results across trials and save the final result
#        (and justifications if needed)
#
#        Args:
#            * results: List of dicts containing the ranking results from trials
#            * justifications: List of justification strings from each trial
#        """
#        # Average the ranking values across trials and save final result
#        rankings = [r["ranking"] for r in results]
#        avg = average_ranking_trials(rankings=rankings)
#        _save_ranking_results(
#            region_1=region_1, region_2=region_2,
#            hemisphere=hemisphere, function=function,
#            model=model, config=config,
#            ranking=avg["mode"],
#            trial="final",
#            analysis_type=analysis_type,
#        )
#
#        # if retesting was done, also save a summary of the retest results
#        if config.retest > 1:
#            func_summary = {
#                "mode": avg["mode"],
#                "counts": avg["counts"],
#                "agreement_ratio": avg[
#                    "agreement_ratio"
#                ],
#                "trials": rankings,
#            }
#            if justifications:
#                func_summary["justifications"] = (
#                    justifications
#                )
#            summary = {
#                "num_trials": config.retest,
#                "functions": {function: func_summary},
#            }
#            _save_retest_summary(
#                region=pair_label, model=model,
#                config=config, analysis_type=analysis_type,
#                hemisphere=hemisphere,
#                summary_data=summary,
#            )
#
#    # Run the task processor with the defined functions
#    process_task(
#        config,
#        model=model,
#        skip_check=skip_check,
#        trial_complete=_trial_done,
#        load_trial=load_trial,
#        prompt_kwargs={
#            "prompt_type": analysis_type,
#            "region_name": pair_label,
#            "hemisphere": hemisphere,
#            "function": function,
#            "region_1": region_1,
#            "region_2": region_2,
#        },
#        log_label=(
#            f"{region_1} vs {region_2} - {function} ({model})"
#        ),
#        process_response=process_response,
#        save_result=save_result,
#        save_final=save_final,
#    )
