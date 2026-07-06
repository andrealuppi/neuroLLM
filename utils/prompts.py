import os
import re
import json
from glob import glob
from typing import List
from pathlib import Path

from utils.misc.logging_setup import logger
from utils.misc.variables import DEFAULT_TEMPLATES, JUSTIFY_OUTPUT_FORMATS
from utils.paths.prompts import PromptPathConstructor


def create_default_templates():
    """Create default template files if they don't exist"""

    # Define two prompting types
    tasks = DEFAULT_TEMPLATES.keys()

    for task in tasks:

        # Ensure prompt directory exists
        default_path = PromptPathConstructor.construct_template_path(
            prompt_type=task, template_name="default"
        )
        os.makedirs(os.path.dirname(default_path), exist_ok=True)

        # Create default template file if missing
        if not os.path.exists(default_path):
            with open(default_path, "w") as f:
                f.write(DEFAULT_TEMPLATES[task])


def get_available_templates(prompt_type: str) -> List[str]:
    """
    Get list of available template names

    Args:
        * prompt_type: "function" or "probability"

    Returns:
        * List of template names (without .txt)
    """
    prompt_dir = PromptPathConstructor.get_prompt_dir(prompt_type=prompt_type)
    pattern = f"{prompt_dir}/*.txt"
    files = glob(pattern)
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_custom_template(prompt_type: str, template_name: str) -> str:
    """
    Load a template from file or return default

    Args:
        * prompt_type: Type of analysis ("top-functions" or "query-functions")
        * template_name: Name of the template file (without .txt)

    Returns:
        * Template string
    """
    create_default_templates()  # Ensure defaults exist

    # Construct template path and ensure it exists
    template_path = PromptPathConstructor.construct_template_path(
        prompt_type=prompt_type, template_name=template_name
    )
    if not os.path.exists(template_path):
        logger.error(
            f"Template {template_name} for {prompt_type} not found",
            exc_info=True,
        )
        raise

    # Load and return template content
    with open(template_path, "r") as f:
        return f.read()


def _apply_justify(prompt: str, prompt_type: str) -> str:
    """
    Replace the Expected Output Format section and remove the
    'DO NOT provide explanations' guideline for justified prompts

    Args:
        * prompt: Formatted prompt string
        * prompt_type: "top-functions", "query-functions", or "rankings"

    Returns:
        * Modified prompt with justify output format
    """
    # Remove "DO NOT provide explanations" guideline
    prompt = re.sub(
        r"\d+\.\s*\*\*DO NOT\*\*\s*provide explanations.*?\n",
        "",
        prompt,
    )

    # Replace Expected Output Format section
    justify_format = JUSTIFY_OUTPUT_FORMATS[prompt_type]
    prompt = re.sub(
        r"### Expected Output Format\s*\n.*",
        justify_format,
        prompt,
        flags=re.DOTALL,
    )

    return prompt


#def generate_prompt(
#    prompt_type: str,
#    region_name: str,
#    species: str,
#    atlas_name: str,
#    hemisphere: str = None,
#    function: str = None,
#    region_1: str = None,
#    region_2: str = None,
#    justify: bool = False,
#    template_name: str = "default",
#    save_to_results: bool = False,
#) -> str:
#    """
#    Generate analysis prompt for functions, probabilities, or rankings
#
#    Args:
#        * prompt_type: "top-functions", "query-functions", or "rankings"
#        * species: Target species
#        * region_name: Brain region name
#        * hemisphere: Hemisphere ("left"/"right"/None)
#        * function: Function name (for probability/ranking prompts)
#        * region_1: First region (for ranking prompts)
#        * region_2: Second region (for ranking prompts)
#        * justify: Whether to include justification request
#        * template_name: Template name to use
#        * atlas_name: Atlas name (for saving)
#        * save_to_results: Whether to save prompt
#
#    Returns:
#        * Generated prompt string
#    """
#    # Load template
#    template = load_custom_template(
#        prompt_type=prompt_type, template_name=template_name
#    )
#
#    # Prepare variables for formatting
#    format_vars = {
#        "species": species,
#        "region": region_name,
#    }
#
#    # Handle hemisphere part
#    format_vars["hemisphere_part"] = (
#        f"in the **{hemisphere} hemisphere** of the"
#        if hemisphere
#        else "in the"
#    )
#
#    # Add function for probability and ranking prompts
#    if prompt_type in ("query-functions", "rankings"):
#        assert function, f"Function must be provided for {prompt_type} prompts"
#        format_vars["function"] = function
#
#    # Add region pair for ranking prompts
#    if prompt_type == "rankings":
#        assert region_1 and region_2, (
#            "region_1 and region_2 must be provided for ranking prompts"
#        )
#        format_vars["region_1"] = region_1
#        format_vars["region_2"] = region_2
#
#    # Format template with all replacements at once
#    prompt = template.format(**format_vars)
#
#    # Apply justify modifications if requested
#    if justify:
#        prompt = _apply_justify(prompt=prompt, prompt_type=prompt_type)
#
#    # Save if requested
#    if save_to_results:
#        save_generated_prompt(
#            prompt=prompt,
#            prompt_type=prompt_type,
#            species=species,
#            region=region_name,
#            hemisphere=hemisphere,
#            atlas_name=atlas_name,
#            template_name=template_name,
#            function=function,
#        )
#
#    return prompt
#

def generate_prompt(
    prompt_type: str,
    region_name: str,
    species: str,
    atlas_name: str,
    hemisphere: str = None,
    function: str = None,
    region_1: str = None,
    region_2: str = None,
    hemisphere_1: str = None,
    hemisphere_2: str = None,
    justify: bool = False,
    template_name: str = "default",
    save_to_results: bool = False,
) -> str:
    """
    Generate analysis prompt for functions, probabilities, or rankings

    Args:
        * prompt_type: "top-functions", "query-functions", or "rankings"
        * species: Target species
        * region_name: Brain region name
        * hemisphere: Hemisphere ("left"/"right"/None/"interhemispheric")
        * function: Function name (for probability/ranking prompts)
        * region_1: First region (for ranking prompts)
        * region_2: Second region (for ranking prompts)
        * hemisphere_1: Hemisphere of first region (for ranking prompts)
        * hemisphere_2: Hemisphere of second region (for ranking prompts)
        * justify: Whether to include justification request
        * template_name: Template name to use
        * atlas_name: Atlas name (for saving)
        * save_to_results: Whether to save prompt

    Returns:
        * Generated prompt string
    """
    # Load template
    template = load_custom_template(
        prompt_type=prompt_type, template_name=template_name
    )

    # Prepare variables for formatting
    format_vars = {
        "species": species,
        "region": region_name,
    }

    # Handle hemisphere part gracefully so "interhemispheric" doesn't sound weird
    # if you ever use {hemisphere_part} in a template again.
    format_vars["hemisphere_part"] = (
        f"in the **{hemisphere} hemisphere** of the"
        if hemisphere and hemisphere != "interhemispheric"
        else "in the"
    )

    # Add function for probability and ranking prompts
    if prompt_type in ("query-functions", "rankings"):
        assert function, f"Function must be provided for {prompt_type} prompts"
        format_vars["function"] = function

    # Add region pair for ranking prompts (ISOLATED FIX)
    if prompt_type == "rankings":
        assert region_1 and region_2, (
            "region_1 and region_2 must be provided for ranking prompts"
        )
        format_vars["region_1"] = region_1
        format_vars["region_2"] = region_2
        
        # Inject hemisphere labels so the template doesn't throw a KeyError
        format_vars["hemisphere_1"] = hemisphere_1 if hemisphere_1 else "Not specified"
        format_vars["hemisphere_2"] = hemisphere_2 if hemisphere_2 else "Not specified"

    # Format template with all replacements at once
    prompt = template.format(**format_vars)

    # Apply justify modifications if requested
    if justify:
        prompt = _apply_justify(prompt=prompt, prompt_type=prompt_type)

    # Save if requested
    if save_to_results:
        save_generated_prompt(
            prompt=prompt,
            prompt_type=prompt_type,
            species=species,
            region=region_name,
            hemisphere=hemisphere,
            atlas_name=atlas_name,
            template_name=template_name,
            function=function,
        )

    return prompt
    
def save_generated_prompt(
    prompt: str,
    prompt_type: str,
    species: str,
    region: str,
    hemisphere: str,
    atlas_name: str,
    template_name: str,
    function: str = None,
):
    """
    Save generated prompt to results

    Args:
        * prompt: Generated prompt string
        * prompt_type: "function" or "probability"
        * species: Species name
        * region: Brain region name
        * hemisphere: Hemisphere ("left"/"right"/None)
        * atlas_name: Atlas name
        * template_name: Template name used
        * function: Function name (for probability prompts)
    """
    # Construct the results prompt path
    prompt_paths = PromptPathConstructor(
        species=species,
        atlas_name=atlas_name,
        hemisphere=hemisphere if hemisphere else "no_separation",
        template_name=template_name,
    )
    results_prompt_path = prompt_paths.construct_results_prompt_path(
        prompt_type=prompt_type,
    )

    # Ensure directory exists
    results_dirname = os.path.dirname(results_prompt_path)
    os.makedirs(results_dirname, exist_ok=True)

    # Check if we already have a prompt saved (avoid duplicates)
    if glob(f"{results_dirname}/prompt_*.txt"):
        return

    # Save new prompt
    with open(results_prompt_path, "w") as f:
        f.write(prompt)

    # Save metadata
    results_timestamp = Path(results_prompt_path).stem.replace("prompt_", "")
    metadata = {
        "template": template_name,
        "species": species,
        "region": region,
        "hemisphere": hemisphere,
        "atlas": atlas_name,
        "function": function,
        "timestamp": results_timestamp,
    }

    # Save metadata alongside prompt
    results_without_ext = os.path.splitext(results_prompt_path)[0]
    with open(f"{results_without_ext}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
