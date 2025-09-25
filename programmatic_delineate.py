import os
import yaml
from delineate import delineate, \
    deep_override  # Import delineate and deep_override functions
from pathlib import Path


def main():
    # Define your configuration here, mirroring the structure expected by delineate.py
    # This example uses values similar to conf_sample.yaml and batch_sample.yaml

    # Ensure output and temp directories exist
    output_dir = Path("data/delineated")
    temp_dir = "/storage/skyterra/tmp"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Load base configuration from conf_sample.yaml
    with open("conf_sample.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    # Define the arguments dictionary that would normally be parsed from CLI or batch config
    input_dir = Path('/storage/skyterra/kz/epsg_32642')
    args = {
        # "input": "data/images/Sample",  # Path to your input images
        "input": input_dir.__str__(),  # Path to your input images
        "output": output_dir / (input_dir.name + '.GPKG'),  # Output GeoPackage file
        "temp": temp_dir,  # Temporary files directory
        "keep_temp": False,  # Whether to keep temporary files
        "mask": None,  # Optional: Path to a mask file (e.g., "data/masks/Sample.tif")
        "config": base_config  # Use the loaded base config
    }

    # You can override specific config values here if needed, similar to batch_routine's "config_override"
    # For example, to change the model:
    args["config"]["model"] = "large"

    # --- Resume/Overwrite Logic ---
    # Set to True to re-process and overwrite the output file if it already exists.
    # By default, it will skip processing if the output file is found.
    overwrite_existing = False

    if not overwrite_existing and os.path.exists(args["output"]):
        print(f"Output file already exists: {args['output']}. Skipping.")
        return

    print(f"Starting delineation for input: {args["input"]}")
    delineate(args, verbose=True)  # verbose=True for more detailed logging
    print(f"Delineation complete. Output saved to: {args["output"]}")


if __name__ == "__main__":
    main()
