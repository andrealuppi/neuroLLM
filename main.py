import sys
import random
from itertools import combinations
from types import SimpleNamespace
from argument_parser import parse_args

import numpy as np
import torch

from dotenv import load_dotenv

from utils.brain_analyser import BrainAnalyser
from utils.api_clients import APIClientManager
from utils.misc.model_listing import list_available_models
from utils.core.function_registry import (
    load_function_group,
    load_functions,
)
from utils.paths.base import DEFAULT_PATHS
from utils.misc.atlas import (
    load_regions_for_species,
    validate_analysis_inputs,
)

from utils.misc.logging_setup import logger


def _resolve_pairs(args, config: SimpleNamespace):
    """
    Resolve pairs for rank-pairs. Uses explicit --pairs if given,
    otherwise generates all combinations from config.regions.

    Args:
        * args: Parsed command-line arguments
        * config: Config object to update with resolved pairs
    """
    # If pairs provided directly, parse and use them
    if args.pairs:
        config.pairs = [
            tuple(p.strip().split(":"))
            for p in args.pairs.split(",")
        ]
        return

    # Generate all region pair combinations
    config.pairs = list(combinations(config.regions, 2))

    # Warn if a large number of pairs to analyze
    n_pairs = len(config.pairs)
    if n_pairs > 100:
        total_queries = (
            n_pairs
            * len(config.functions)
            * len(config.models)
            * config.retest
        )
        logger.warning_status(
            f"Generating {n_pairs} pairs ({total_queries} total queries). "
            "This may take a while."
        )


def _resolve_functions(args) -> list:
    """
    Resolve functions from --function-group, --functions, or defaults

    Args:
        * args: Parsed command-line arguments

    Returns:
        * List of functions to analyze
    """

    # Priority: function group > specified functions > default functions
    if args.function_group:
        # Load functions from the specified group
        functions = load_function_group(
            group_name=args.function_group
        )
        # Validate that the group was found and has functions
        if not functions:
            logger.error_status(
                f"Function group '{args.function_group}' not found."
            )
            sys.exit(1)
        logger.info(
            f"Using functions from group '{args.function_group}': "
            f"{', '.join(functions)}"
        )

    elif args.functions:
        # Parse functions from the comma-separated list
        functions = [
            f.strip() for f in args.functions.split(",")
        ]
        logger.info(
            f"Using specified functions: {', '.join(functions)}"
        )

    else:
        # Load default functions if no group or specific functions provided
        functions, _ = load_functions()
        logger.info(
            f"Using default functions: {', '.join(functions)}"
        )

    return functions


def main():
    """Main LLM brain analysis script"""
    # Set global seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load environment variables from .env file
    load_dotenv(DEFAULT_PATHS["env_file"])

    # Parse command-line arguments
    args = parse_args()

    # Handle list-models command early exit
    if args.command == "list-models":
        list_available_models(filter=args.filter)
        sys.exit(0)

    # Determine models to use based on command
    models_input = "dummy" if args.command == "test" else args.models

    # Validate we have sufficient region/atlas information to proceed
    validate_analysis_inputs(args=args)

    # Load regions once: from --regions if given, else from atlas
    regions = (
        [r.strip() for r in args.regions.split(",")]
        if args.regions
        else None
    )
    if regions is None and args.atlas_name:
        regions = load_regions_for_species(
            species=args.species,
            atlas_name=args.atlas_name,
        )

    # Resolve max_tokens default based on --justify
    # Resolve max_tokens default based on --justify and local models
    if args.max_tokens is None:
        has_local_model = any(m.strip().startswith("local/") for m in args.models.split(","))
        if has_local_model:
            args.max_tokens = 256 if args.justify else 128
        else:
            args.max_tokens = 512 if args.justify else 256
        
    # Initialize API clients
    logger.info("Initializing API clients...")
    try:
        client_manager = APIClientManager(
            models=models_input,
            embedding_provider=(
                args.embedding_provider
                if args.command == "top-functions"
                else None
            ),
            max_tokens=args.max_tokens,
        )
        model_names = client_manager.model_names
        logger.info(f"Using models: {', '.join(model_names)}")
    except Exception as e:
        logger.error_status(
            f"Failed to initialize API clients: {e}", exc_info=True
        )
        sys.exit(1)

    # Create base config as SimpleNamespace
    config = SimpleNamespace(
        species=args.species,
        regions=regions,
        models=model_names,
        functions=None,  # Set later for probabilities/rankings
        pairs=None,  # Set later for rankings
        workers=(
            1
            if args.command == "test"
            or any(m.startswith("local/") for m in model_names)
            else args.workers
        ),
        skip_visualization=args.skip_visualization,
        skip_raw_saving=args.skip_raw_saving,
        atlas_name=args.atlas_name,
        separate_hemispheres=args.separate_hemispheres,
        prompt_template_name=args.prompt_template_name,
        client_manager=client_manager,
        justify=args.justify,
        retest=args.retest,
        temperature=args.temperature,
        consensus_threshold=getattr(args, "consensus_threshold", 0.80),
    )

    # Run the appropriate command
    try:
        if args.command == "top-functions":
            logger.info(f"Running functions analysis for {args.species}...")
            analyser = BrainAnalyser(config=config)
            analyser.analyze_functions()

        elif args.command == "query-functions":
            # Resolve functions to analyze based on arguments
            config.functions = _resolve_functions(args=args)
            logger.info(
                f"Running probabilities analysis for {args.species}..."
            )
            analyser = BrainAnalyser(config=config)
            analyser.analyze_probabilities()

        elif args.command == "rank-pairs":
            # Resolve functions and pairs to analyze based on arguments
            config.functions = _resolve_functions(args=args)
            _resolve_pairs(args=args, config=config)
            logger.info(
                f"Running rankings analysis for {args.species}..."
            )
            analyser = BrainAnalyser(config=config)
            analyser.analyze_rankings()

        elif args.command == "test":
            logger.info("Running test workflow...")
            analyser1 = BrainAnalyser(config=config)

            logger.info("Testing functions analysis...")
            analyser1.analyze_functions()
            logger.success("Functions test completed")

            # Test probability analysis with dummy model
            logger.info("Testing probabilities analysis...")
            config.functions = [
                "spatial cognition",
                "consciousness",
            ]

            analyser2 = BrainAnalyser(config=config)
            analyser2.analyze_probabilities()
            logger.success("Probabilities test completed")

            # Test ranking analysis with dummy model
            logger.info("Testing rankings analysis...")
            test_regions = config.regions[:3]
            config.pairs = list(combinations(test_regions, 2))

            analyser3 = BrainAnalyser(config=config)
            analyser3.analyze_rankings()
            logger.success("Rankings test completed")

        else:
            logger.error_status(
                f"Unknown command: {args.command}, please use from available "
                "commands: list-models, top-functions, query-functions, "
                "rank-pairs, test"
            )
            sys.exit(1)

        logger.success("Analysis completed successfully!")

    except KeyboardInterrupt:
        logger.error("\nAnalysis interrupted by user", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error_status(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
