
# ScanFastLabs

## Overview
ScanFast is a comprehensive toolkit for image classification and report generation using deep learning models. It provides a streamlined process for utilizing pre-trained models on medical datasets, with optional language model integration for enhanced report generation.

## Installation
Ensure you have Anaconda installed to manage your Python environment.

```bash
conda create -n scan_fast python=3.11.5
conda activate scan_fast
conda install pytorch::pytorch torchvision -c pytorch
pip install reportlab timm matplotlib medmnist langchain llama-cpp-python
git clone https://github.com/ScanFast-Labs/ScanFast-Local.git
cd ScanFast-Local
```

## Setup
1. Download required models and images:
   - Image Classification Model & Test Images: [Google Drive Link](https://drive.google.com/drive/folders/175XpesccbmsnUXykBEhoAxTV5ZRabpR_?usp=share_link)
   - LLM Model (Optional for full reports): [Hugging Face Link](https://huggingface.co/TheBloke/meditron-7B-GGUF)
     - For 16+ GB RAM: Q8 is recommended for the best performance.
     - For 8 GB RAM: Q3 or Q4 will work.
2. Set file paths in `config.yaml` for the downloaded models and images.

## Usage
- For image classification only:
  ```bash
  python3 main.py image-only
  ```
- To generate a full report:
  ```bash
  python3 main.py full-scan
  ```

## Repository Structure
- `config_loader.py`: Loads configuration settings from `config.yaml`.
- `image_classification.py`: Handles the image classification process.
- `report.py`: Manages the generation of reports post-classification.
- `main.py`: The main script to run the program (either image classification or full report generation).
- `model_utils.py`: Utility functions for model handling and processing.
- `config.yaml`: Configuration file to set paths and parameters.
- `models.py`: Defines the deep learning models used in the toolkit.
- `data_utils.py`: Utilities for data handling and preprocessing.


