# Delineate Anything: Resolution-Agnostic Field Boundary Delineation

## Project Overview

This project, "Delineate Anything," is a deep learning framework for accurately delineating agricultural field boundaries from satellite imagery. It is designed to be resolution-agnostic, meaning it can work with satellite images of varying resolutions. The project is written in Python and utilizes the PyTorch library for deep learning, specifically leveraging the Ultralytics library. The models are hosted on Hugging Face, and the environment is managed using Conda and pip.

The core logic is within the `delineate.py` script, which can be run in a single-image or batch mode. The delineation process is highly configurable through YAML files, allowing users to specify models, data loading parameters, and post-processing steps.

## Building and Running

### Environment Setup

To set up the environment, you can use either Conda or pip. The `README.md` provides detailed instructions for both Linux and Windows systems.

**Linux:**

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda install -c conda-forge gdal

pip install torch==2.6.0
pip install -r requirements.txt
```

**Windows:**

```bash
conda create --prefix=./.conda python=3.11
conda activate ./.conda
conda install -c conda-forge gdal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Running Inference

The main script for running the delineation is `delineate.py`. It can be run in batch mode by providing a batch configuration file.

1.  **Place your images:** Put your RGB images in the `data/images/` folder. If you have land cover maps, place them in the `data/masks/` folder.
2.  **Configure your run:** Create a YAML configuration file for your run. You can use `conf_sample.yaml` as a template for a single run and `batch_sample.yaml` for a batch run.
3.  **Run the script:**

    ```bash
    python delineate.py -b batch_sample.yaml
    ```

The output, a GeoPackage file with the delineated field boundaries, will be saved in the `data/delineated/` directory.

## Development Conventions

*   **Configuration:** The project heavily relies on YAML files for configuration. The `conf_sample.yaml` and `batch_sample.yaml` files provide a good starting point for understanding the available options.
*   **Dependencies:** Project dependencies are managed in `requirements.txt`.
*   **Models:** Pre-trained models are downloaded from Hugging Face. The `delineate.py` script handles the download and caching of these models.
*   **Extensibility:** The project is designed to be extensible. New delineation methods can be added by creating a new subfolder in the `methods` directory and implementing the required `execute` function in an `inference.py` file.
*   **Entry Point:** The main entry point for the application is `delineate.py`.
