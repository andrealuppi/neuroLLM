import os
import json
import pandas as pd
from typing import Dict, Any

from utils.misc.logging_setup import logger

from utils.paths.query import QueryPathConstructor
from utils.paths.embeddings import EmbeddingsPathConstructor
from utils.paths.aggregation import AggregatedResultsPathConstructor


def _iter_model_hemispheres(config, analysis_type, with_embeddings=False):
    """
    Yield (model, hemisphere, query, agg, emb) for each model/hemisphere
    combination

    Args:
        * config: Analysis configuration
        * analysis_type: Analysis type string
        * with_embeddings: If True, also construct EmbeddingsPathConstructor
    """
#    hemispheres = (
#        ["left", "right"]
#        if config.separate_hemispheres
#        else [None]
#    )

    if config.separate_hemispheres:
        hemispheres = ["left", "right"]
        # Only include the interhemispheric folder for pairwise analyses
        if analysis_type == "rankings":
            hemispheres.append("interhemispheric")
    else:
        hemispheres = [None]
        
    for hemisphere in hemispheres:
        for model in config.models:
            common = dict(
                model=model, species=config.species,
                atlas_name=config.atlas_name,
                analysis_type=analysis_type,
                hemisphere=hemisphere,
                template_name=config.prompt_template_name,
            )
            query = QueryPathConstructor(**common)
            agg = AggregatedResultsPathConstructor(**common)
            emb = (
                EmbeddingsPathConstructor(**common)
                if with_embeddings else None
            )
            yield model, hemisphere, query, agg, emb


def aggregate_function_results(
    config: Dict[str, Any], analysis_type: str = "top-functions"
):
    """
    Function aggregation. Creates:
    - results/aggregated/functions/{species}/{atlas}/{model}/{template}/
        {hemisphere}/all_responses.json
    - results/aggregated/functions/{species}/{atlas}/{model}/{template}/
        {hemisphere}/all_embeddings.csv

    Args:
        * config: Analysis configuration
        * analysis_type: "top-functions" or "query-functions"
    """
    for model, hemisphere, query, agg, emb in _iter_model_hemispheres(
        config, analysis_type, with_embeddings=True,
    ):
        # Collect all function results for this model/hemisphere
        all_responses = {}
        embedding_dfs = []
        per_func_embedding_dfs = []

        for region in config.regions:

            # Get the function response path
            res_path = query.construct_query_cleaned_region_path(
                region=region, trial="final",
            )
            if not os.path.exists(res_path):
                logger.error_status(
                    f"No query results file found: {res_path}",
                    exc_info=True,
                )
                raise

            # Load function response
            with open(res_path) as f:
                data = json.load(f)
                if model in data:
                    all_responses[region] = data[model]

            # Get the combined embedding path
            emb_path = emb.construct_embeddings_region_path(
                region=region, trial="final",
            )
            if not os.path.exists(emb_path):
                logger.error_status(
                    f"No embeddings file found: {emb_path}",
                    exc_info=True,
                )
                raise

            # Load combined embedding
            df = pd.read_csv(emb_path, index_col=0)
            embedding_dfs.append(df)

            # Load per-function embeddings if available
            pf_path = emb.construct_per_function_embeddings_region_path(
                region=region, trial="final",
            )
            if os.path.exists(pf_path):
                pf_df = pd.read_csv(pf_path, index_col=0)
                # Add region column for multi-index
                pf_df.insert(0, "region", region)
                per_func_embedding_dfs.append(pf_df)

        # Get aggregated paths
        agg_path = agg.construct_aggregated_query_results_path()
        aggemb_path = agg.construct_aggregated_embeddings_path()
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        os.makedirs(os.path.dirname(aggemb_path), exist_ok=True)

        # Save query responses
        with open(agg_path, "w") as f:
            json.dump(all_responses, f, indent=2)
        logger.processing(
            f"Saved {len(all_responses)} function responses for "
            f"{model}/{hemisphere if hemisphere else 'no_separation'}"
        )

        # Save combined embeddings as CSV
        all_embeddings_df = pd.concat(embedding_dfs)
        all_embeddings_df.to_csv(aggemb_path)
        logger.processing(
            f"Saved {len(embedding_dfs)} embeddings for "
            f"{model}/{hemisphere if hemisphere else 'no_separation'}"
        )

        # Save per-function embeddings if available
        if per_func_embedding_dfs:
            aggpf_path = (
                agg.construct_aggregated_per_function_embeddings_path()
            )
            os.makedirs(os.path.dirname(aggpf_path), exist_ok=True)
            all_pf_df = pd.concat(per_func_embedding_dfs)
            all_pf_df.to_csv(aggpf_path)
            logger.processing(
                f"Saved per-function embeddings for "
                f"{model}/{hemisphere if hemisphere else 'no_separation'}"
            )


def aggregate_probability_results(
    config: Dict[str, Any], analysis_type: str = "query-functions"
):
    """
    Probability aggregation. Creates:
    - results/aggregated/probabilities/{species}/{atlas}/{model}/{template}/
        {hemisphere}/probability_distribution.csv
    - results/aggregated/probabilities/{species}/{atlas}/{model}/{template}/
        {hemisphere}/{function}/probabilities.csv

    Args:
        * config: Analysis configuration
        * analysis_type: "top-functions" or "query-functions"
    """
    for model, hemisphere, query, agg, _ in _iter_model_hemispheres(
        config, analysis_type,
    ):
        # Collect probability data
        all_data = {}

        for region in config.regions:

            # Get the function response path
            res_path = query.construct_query_region_path(
                region=region, trial="final",
            )
            if not os.path.exists(res_path):
                logger.error_status(
                    f"No query results file found: {res_path}",
                    exc_info=True,
                )
                raise

            # Load function response
            with open(res_path) as f:
                region_data = json.load(f)
                all_data[region] = region_data

        # Create overview DataFrame
        df_data = []
        for region in config.regions:
            if region in all_data:
                row = {"Region": region}
                for function in config.functions:
                    # Cleaner probability extraction
                    prob_value = (
                        all_data[region].get(function, {}).get(model)
                    )
                    row[function] = prob_value
                df_data.append(row)

        df = pd.DataFrame(df_data).set_index("Region")

        # Get aggregated paths
        agg_path = agg.construct_aggregated_query_results_path(
            extension="csv",
        )
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        df.to_csv(agg_path)
        logger.processing(
            f"Saved probability overview for {model}/"
            f"{hemisphere if hemisphere else 'no_separation'}"
        )

        # Save individual function files
        for function in config.functions:
            path = agg.construct_individual_function_prob_path(
                function=function,
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Extract function probabilities and drop NaNs
            func_df = (
                df[[function]]
                .dropna()
                .rename(columns={function: "Probability"})
            )

            # Save if not empty
            if not func_df.empty:
                func_df.to_csv(path)
                logger.processing(
                    f"Saved {len(func_df)} probabilities for {function}"
                )


#def aggregate_ranking_results(
#    config: Dict[str, Any], analysis_type: str = "rankings"
#):
#    """
#    Ranking aggregation. Creates one CSV per pair showing
#    which region won for each function.
#
#    Output per pair:
#    - {region1}_vs_{region2}/results.csv
#      Columns: Function, Winner, Ranking
#
#    Args:
#        * config: Analysis configuration
#        * analysis_type: "rankings"
#    """
#    for model, _, query, agg, _ in _iter_model_hemispheres(
#        config, analysis_type,
#    ):
#        for r1, r2 in config.pairs:
#            rows = []
#            for function in config.functions:
#                path = query.construct_query_pair_path(
#                    region_1=r1, region_2=r2,
#                    trial="final",
#                )
#                if not os.path.exists(path):
#                    continue
#
#                with open(path) as f:
#                    data = json.load(f)
#
#                ranking = (
#                    data.get(function, {}).get(model)
#                )
#                if ranking is None:
#                    continue
#
#                winner = r1 if ranking == 1 else r2
#                rows.append({
#                    "Function": function,
#                    "Winner": winner,
#                })
#
#            if not rows:
#                continue
#
#            df = pd.DataFrame(rows).set_index("Function")
#            agg_path = agg.construct_aggregated_pair_results_path(
#                region_1=r1, region_2=r2,
#            )
#            os.makedirs(
#                os.path.dirname(agg_path), exist_ok=True
#            )
#            df.to_csv(agg_path)
#
#            logger.processing(
#                f"Saved pair results for "
#                f"{r1} vs {r2} ({model})"
#            )

#def aggregate_ranking_results(
#    config: Dict[str, Any], analysis_type: str = "rankings"
#):
#    """
#    Ranking aggregation. Creates one CSV per function showing
#    which region won for each pair.
#    
#    Output per function:
#    - {function}/rankings.csv
#      Columns: Pair, Region 1, Region 2, Winner
#    """
#    for model, _, query, agg, _ in _iter_model_hemispheres(
#        config, analysis_type,
#    ):
#        for function in config.functions:
#            rows = []
#            for r1, r2 in config.pairs:
#                path = query.construct_query_pair_path(
#                    region_1=r1, region_2=r2,
#                    trial="final",
#                )
#                if not os.path.exists(path):
#                    continue
#
#                with open(path) as f:
#                    data = json.load(f)
#
#                ranking = data.get(function, {}).get(model)
#                if ranking is None:
#                    continue
#
#                winner = r1 if ranking == 1 else r2
#                rows.append({
#                    "Pair": f"{r1}_vs_{r2}",
#                    "Region 1": r1,
#                    "Region 2": r2,
#                    "Winner": winner,
#                })
#
#            if not rows:
#                continue
#
#            # Create one dataframe for the entire function
#            df = pd.DataFrame(rows).set_index("Pair")
#            
#            # Trick to get the root directory by generating a dummy pair path
#            dummy_path = agg.construct_aggregated_pair_results_path(
#                region_1="A", region_2="B",
#            )
#            root_dir = os.path.dirname(os.path.dirname(dummy_path))
#            
#            # Construct the new function-specific path
#            agg_path = os.path.join(root_dir, function, "rankings.csv")
#            
#            os.makedirs(os.path.dirname(agg_path), exist_ok=True)
#            df.to_csv(agg_path)
#
#            logger.processing(
#                f"Saved ranking results for {function} ({model})"
#            )

def aggregate_ranking_results(
    config: Dict[str, Any], analysis_type: str = "rankings"
):
    """
    Ranking aggregation. Creates one CSV per function showing
    which region won for each pair.
    
    Output per function:
    - {function}/rankings.csv
      Columns: Pair, Region 1, Region 2, Winner
    """
    for model, hemisphere, query, agg, _ in _iter_model_hemispheres(
        config, analysis_type,
    ):
        for function in config.functions:
            rows = []
            
            # Extract unique regions dynamically
            unique_regions = list(set([r for pair in config.pairs for r in pair]))
            
            # Determine pairs based on hemisphere category
            if hemisphere == "interhemispheric":
                current_pairs = [(r1, r2) for r1 in unique_regions for r2 in unique_regions]
            else:
                current_pairs = list(config.pairs)

            for r1, r2 in current_pairs:
                path = query.construct_query_pair_path(
                    region_1=r1, region_2=r2,
                    trial="final",
                )
                if not os.path.exists(path):
                    continue

                with open(path) as f:
                    data = json.load(f)

                ranking = data.get(function, {}).get(model)
                if ranking is None:
                    continue

                # --- NEW LOGIC: Format display names based on hemisphere ---
                if hemisphere == "left":
                    r1_disp, r2_disp = f"left {r1}", f"left {r2}"
                elif hemisphere == "right":
                    r1_disp, r2_disp = f"right {r1}", f"right {r2}"
                elif hemisphere == "interhemispheric":
                    r1_disp, r2_disp = f"left {r1}", f"right {r2}"
                else:
                    # Unseparated (None) fallback
                    r1_disp, r2_disp = r1, r2

                # Determine winner using the new display names
                winner_disp = r1_disp if ranking == 1 else r2_disp
                
                rows.append({
                    "Pair": f"{r1_disp}_vs_{r2_disp}",
                    "Region 1": r1_disp,
                    "Region 2": r2_disp,
                    "Winner": winner_disp,
                })

            if not rows:
                continue

            # Create one dataframe for the entire function
            df = pd.DataFrame(rows).set_index("Pair")
            
            # Trick to get the root directory by generating a dummy pair path
            dummy_path = agg.construct_aggregated_pair_results_path(
                region_1="A", region_2="B",
            )
            root_dir = os.path.dirname(os.path.dirname(dummy_path))
            
            # Construct the new function-specific path
            agg_path = os.path.join(root_dir, function, "rankings.csv")
            
            os.makedirs(os.path.dirname(agg_path), exist_ok=True)
            df.to_csv(agg_path)

            logger.processing(
                f"Saved ranking results for {function} ({model})"
            )
            
def aggregate_justifications(
    config: Dict[str, Any], analysis_type: str
):
    """
    Aggregate justification files into a single JSON per
    model/hemisphere

    Args:
        * config: Analysis configuration
        * analysis_type: "top-functions", "query-functions", or
            "rankings"
    """
    for model, hemisphere, query, agg, _ in _iter_model_hemispheres(
        config, analysis_type,
    ):
        all_justifications = {}

        if analysis_type == "rankings":
            # Aggregate pair justifications
            for r1, r2 in config.pairs:
                just_path = (
                    query
                    .construct_query_pair_justification_path(
                        region_1=r1, region_2=r2,
                        trial="final",
                    )
                )
                if os.path.exists(just_path):
                    with open(just_path) as f:
                        data = json.load(f)
                    pair_key = f"{r1}_vs_{r2}"
                    all_justifications[pair_key] = data
        else:
            # Aggregate region justifications
            for region in config.regions:
                just_path = (
                    query
                    .construct_query_justification_region_path(
                        region=region, trial="final",
                    )
                )
                if os.path.exists(just_path):
                    with open(just_path) as f:
                        data = json.load(f)
                    all_justifications[region] = data

        if all_justifications:
            agg_path = (
                agg.construct_aggregated_justification_path()
            )
            os.makedirs(
                os.path.dirname(agg_path), exist_ok=True
            )
            with open(agg_path, "w") as f:
                json.dump(
                    all_justifications, f, indent=2
                )
            logger.processing(
                f"Saved justification aggregation for "
                f"{model}/{hemisphere or 'no_separation'}"
            )


def aggregate_retest_statistics(
    config: Dict[str, Any], analysis_type: str
):
    """
    Aggregate retest summary files into overview CSVs

    Args:
        * config: Analysis configuration
        * analysis_type: "top-functions", "query-functions", or
            "rankings"
    """
    for model, hemisphere, query, agg, _ in _iter_model_hemispheres(
        config, analysis_type,
    ):
        if analysis_type == "top-functions":
            # Consistency scores per region
            scores = {}
            for region in config.regions:
                path = query.construct_query_retest_summary_path(
                    region=region,
                )
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    if model in data:
                        scores[region] = data[model].get(
                            "consistency_score", None
                        )

            if scores:
                df = pd.DataFrame.from_dict(
                    scores, orient="index",
                    columns=["consistency_score"],
                )
                df.index.name = "Region"
                out = agg.construct_aggregated_retest_stats_path(
                    filename="consistency_scores.csv",
                )
                os.makedirs(
                    os.path.dirname(out), exist_ok=True
                )
                df.to_csv(out)

#        elif analysis_type == "query-functions":
#            # Mean + std per region x function
#            rows = []
#            for region in config.regions:
#                path = query.construct_query_retest_summary_path(
#                    region=region,
#                )
#                if not os.path.exists(path):
#                    continue
#                with open(path) as f:
#                    data = json.load(f)
#                if model not in data:
#                    continue
#                row = {"Region": region}
#                funcs = data[model].get("functions", {})
#                for func_name, stats in funcs.items():
#                    row[f"{func_name}_mean"] = (
#                        stats.get("mean")
#                    )
#                    row[f"{func_name}_std"] = (
#                        stats.get("std")
#                    )
#                    row[f"{func_name}_min"] = (
#                        stats.get("min")
#                    )
#                    row[f"{func_name}_max"] = (
#                        stats.get("max")
#                    )
#                rows.append(row)
#
#            if rows:
#                df = pd.DataFrame(rows).set_index("Region")
#                out = agg.construct_aggregated_retest_stats_path(
#                    filename="retest_statistics.csv",
#                )
#                os.makedirs(
#                    os.path.dirname(out), exist_ok=True
#                )
#                df.to_csv(out)


        elif analysis_type == "query-functions":
            # Mean + std per region, saved into specific function folders
            for function in config.functions:
                rows = []
                for region in config.regions:
                    path = query.construct_query_retest_summary_path(
                        region=region,
                    )
                    if not os.path.exists(path):
                        continue
                    with open(path) as f:
                        data = json.load(f)
                    if model not in data:
                        continue
                    
                    # Only pull the stats for the current function in the loop
                    stats = data[model].get("functions", {}).get(function)
                    if stats:
                        rows.append({
                            "Region": region,
                            "mean": stats.get("mean"),
                            "std": stats.get("std"),
                            "min": stats.get("min"),
                            "max": stats.get("max")
                        })

                if rows:
                    df = pd.DataFrame(rows).set_index("Region")
                    
                    # Trick to get the exact folder path where probabilities.csv lives
                    prob_path = agg.construct_individual_function_prob_path(
                        function=function
                    )
                    out = os.path.join(os.path.dirname(prob_path), "retest_statistics.csv")
                    
                    os.makedirs(
                        os.path.dirname(out), exist_ok=True
                    )
                    df.to_csv(out)

#        elif analysis_type == "rankings":
#            # Agreement ratios per pair x function
#            rows = []
#            for r1, r2 in config.pairs:
#                pair_name = f"{r1}_vs_{r2}"
#                path = query.construct_query_retest_summary_path(
#                    region=pair_name,
#                )
#                if not os.path.exists(path):
#                    continue
#                with open(path) as f:
#                    data = json.load(f)
#                if model not in data:
#                    continue
#                row = {"Pair": pair_name}
#                funcs = data[model].get(
#                    "functions", {}
#                )
#                for func_name, stats in funcs.items():
#                    row[func_name] = stats.get(
#                        "agreement_ratio"
#                    )
#                rows.append(row)
#
#            if rows:
#                df = pd.DataFrame(rows).set_index("Pair")
#                out = agg.construct_aggregated_retest_stats_path(
#                    filename="retest_agreement.csv",
#                )
#                os.makedirs(
#                    os.path.dirname(out), exist_ok=True
#                )
#                df.to_csv(out)

        elif analysis_type == "rankings":
            # Agreement ratios per pair, saved into specific function folders
            for function in config.functions:
                rows = []
                for r1, r2 in config.pairs:
                    pair_name = f"{r1}_vs_{r2}"
                    path = query.construct_query_retest_summary_path(
                        region=pair_name,
                    )
                    if not os.path.exists(path):
                        continue
                    with open(path) as f:
                        data = json.load(f)
                    if model not in data:
                        continue
                    
                    # Only grab the agreement ratio for the current function
                    stats = data[model].get("functions", {}).get(function)
                    if stats:
                        rows.append({
                            "Pair": pair_name,
                            "agreement_ratio": stats.get("agreement_ratio")
                        })

                if rows:
                    df = pd.DataFrame(rows).set_index("Pair")
                    
                    # Trick to get the root directory using a dummy filename
                    dummy_out = agg.construct_aggregated_retest_stats_path(
                        filename="dummy.csv",
                    )
                    
                    # Save to: .../{function}/retest_agreement.csv
                    out = os.path.join(os.path.dirname(dummy_out), function, "retest_agreement.csv")
                    
                    os.makedirs(
                        os.path.dirname(out), exist_ok=True
                    )
                    df.to_csv(out)

        logger.processing(
            f"Saved retest statistics for {model}/"
            f"{hemisphere or 'no_separation'}"
        )


def aggregate_results(config: Dict[str, Any], analysis_type: str):
    """
    Aggregate results based on analysis type

    Args:
        * config: Analysis configuration
        * analysis_type: "top-functions", "query-functions", or
            "rankings"
    """
    if analysis_type == "top-functions":
        aggregate_function_results(
            config=config, analysis_type=analysis_type
        )
    elif analysis_type == "query-functions":
        aggregate_probability_results(
            config=config, analysis_type=analysis_type
        )
    elif analysis_type == "rankings":
        aggregate_ranking_results(
            config=config, analysis_type=analysis_type
        )
    else:
        raise ValueError(
            f"Unknown analysis type: {analysis_type}"
        )

    # Aggregate justifications if --justify was used
    if config.justify:
        aggregate_justifications(
            config=config, analysis_type=analysis_type
        )

    # Aggregate retest statistics if --retest > 1
    if config.retest > 1:
        aggregate_retest_statistics(
            config=config, analysis_type=analysis_type
        )

    logger.success("Aggregation completed successfully!")
