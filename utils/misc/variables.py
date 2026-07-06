DEFAULT_FUNCTION_TEMPLATE = """
    You are an expert in neuroscience literature analysis. Your task is to
    list out the top 5 functions that region region **{region}**
    {hemisphere_part} **{species}** brain is involved in.

    These functions should be based on a simulated review of neuroscience
    literature, reflecting how frequently these functions are associated with
    this brain region across **peer-reviewed** studies, textbooks, and
    reputable sources.

    ### Guidelines
    1. Consider only **{species}**-specific neuroscience literature. Do **not**
    use data from other species.
    2. **DO NOT** provide explanations, citations, or any extra text—**return
    only the function names**.
    3. Return your results as a list: [function_1, function_2, function_3,
    function_4, function_5]
    4. **DO NOT** repeat functions in your list.
    5. List out the functions **only** for the specified {hemisphere_part}
    hemisphere.

    ### Expected Output Format
    [function_1, function_2, function_3, function_4, function_5]
    """

DEFAULT_PROBABILITY_TEMPLATE = """
    You are an expert in neuroscience literature analysis. Your task is to
    estimate the probability that the brain region region **{region}**
    {hemisphere_part} **{species}** brain is involved in **
    {function}**.

    These functions should be based on a simulated review of neuroscience
    literature, reflecting how frequently these functions are associated with
    this brain region across **peer-reviewed** studies, textbooks, and
    reputable sources.

    ### Guidelines
    1. Consider only **{species}**-specific neuroscience literature. Do
    **not** use data from other species.
    2. The probability should be a **single decimal number** between **0 and
    1**.
    3. This number should **approximate the relative frequency** with which
    this function is linked to the given brain region in literature.
    4. **DO NOT** provide explanations, citations, or any extra text—**return
    only the probability value**.

    ### Expected Output Format
    0.XX
    """


DEFAULT_RANKING_TEMPLATE = """
    You are an expert in neuroscience literature analysis. Your task is to
    determine which of two brain regions is more relevant to the function
    **{function}** in the {hemisphere_part} **{species}** brain.

    Region 1: **{region_1}**
    Region 2: **{region_2}**

    Base your assessment on a simulated review of neuroscience literature,
    reflecting how frequently each region is associated with this function
    across **peer-reviewed** studies, textbooks, and reputable sources.

    ### Guidelines
    1. Consider only **{species}**-specific neuroscience literature. Do **not**
    use data from other species.
    2. **DO NOT** provide explanations, citations, or any extra text—**return
    only the number**.
    3. Return **1** if Region 1 is more relevant, or **2** if Region 2 is
    more relevant.

    ### Expected Output Format
    1 or 2
    """

DEFAULT_TEMPLATES = {
    "top-functions": DEFAULT_FUNCTION_TEMPLATE,
    "query-functions": DEFAULT_PROBABILITY_TEMPLATE,
    "rankings": DEFAULT_RANKING_TEMPLATE,
}

JUSTIFY_OUTPUT_FORMATS = {
    "top-functions": (
        "### Expected Output Format\n"
        "    [function_1, function_2, function_3, function_4, function_5]"
        " | {justification}\n\n"
        "    Where {justification} is a 1-2 sentence explanation (max 200 "
        "characters) of why these are the top functions for this region. "
        'Use " | " (space-pipe-space) as the separator between the list '
        "and your justification."
    ),
    "query-functions": (
        "### Expected Output Format\n"
        "    0.XX | {justification}\n\n"
        "    Where {justification} is a 1-2 sentence explanation (max 200 "
        "characters) of your probability estimate. "
        'Use " | " (space-pipe-space) as the separator.'
    ),
    "rankings": (
        "### Expected Output Format\n"
        "    1 | {justification} or 2 | {justification}\n\n"
        "    Where {justification} is a 1-2 sentence explanation (max 200 "
        "characters) of your ranking. "
        'Use " | " (space-pipe-space) as the separator.'
    ),
}


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#BRAINGPT_CONFIG = {
#    "base_model_id": "meta-llama/Llama-2-7b-chat-hf",
#    "adapter_id": "BrainGPT/BrainGPT-7B-v0.1",
#}

LOCAL_MODEL_PREFIX = "local/"

#LOCAL_MODEL_REGISTRY = {
#    "local/medgemma": {
#        "backend": "mlx",
#        "path": "mlx-community/medgemma-4b-it-4bit",
#    },
#    "local/braingpt": {
#        "backend": "braingpt",
#    },
#}

#LOCAL_MODELS = set(LOCAL_MODEL_REGISTRY.keys()) | {"dummy"}

import json
from pathlib import Path

LOCAL_MODEL_CONFIG_PATH = Path("config/local_models.json")

if LOCAL_MODEL_CONFIG_PATH.exists():
    with open(LOCAL_MODEL_CONFIG_PATH, "r") as f:
        LOCAL_MODEL_REGISTRY = json.load(f)
else:
    LOCAL_MODEL_REGISTRY = {}

LOCAL_MODELS = set(LOCAL_MODEL_REGISTRY.keys()) | {"dummy"}

EMBEDDING_DIMS = {"openai": 3072, "local": 1024}

DEFAULT_FUNCTIONS = [
    "cognitive control",
    "emotion",
    "language",
    "listening",
    "manipulation",
    "memory",
    "reward",
    "vision",
]
