import argparse
import json
import sys
import logging
from pathlib import Path
import traceback

from pipelines.training_pipeline import TrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(
        description="Run the vehicle dynamics model training pipeline.",
        epilog="Example: python scripts/train_model.py -c configs/training/experiment_1.json"
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Path to the training configuration JSON file."
    )
    
    # Force help menu if run blindly
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    
    if not config_path.exists():
        logging.error(f"[Error] Configuration file not found at {config_path}")
        sys.exit(1)

    logging.info(f"Loading training configuration from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Force absolute paths to prevent CWD bugs
    project_root = Path(__file__).resolve().parent.parent
    if "dataset" in config and "path" in config["dataset"]:
        config["dataset"]["path"] = str(project_root / config["dataset"]["path"])
            
    try:
        pipeline = TrainingPipeline(config=config)
        pipeline.run()
    except FileNotFoundError as e:
        logging.error(f"\n[I/O Error] {e}")
        sys.exit(1)
    except KeyError as e:
        logging.error(f"\n[Config Error] Missing required key in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n[Pipeline Error] An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()