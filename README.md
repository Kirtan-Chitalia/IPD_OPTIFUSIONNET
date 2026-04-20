# OptifusionNet

OptiFusionNet is a deep learning model designed for image enhancement using a two-resolution training strategy to achieve both speed and high quality. It combines meta-optimization techniques (PSO and ACO) for efficient hyperparameter search with a multi-stage training pipeline (fast Adam pre-training, high-quality Adam training, and final SGD fine-tuning).

## Features
- **Two-Resolution Training**: Starts with a fast, lower-resolution training stage and transitions to a high-quality, higher-resolution stage.
- **Meta-Optimization**: Utilizes Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO) to find optimal learning rates, momentum, and weight decay.
- **Hybrid Loss Function**: Combines L1 loss with a gradient loss component for improved image quality.
- **Residual Denoise Blocks**: Incorporates residual denoise blocks for effective noise reduction and feature learning.
- **Flexible Dataset Handling**: Supports various image formats (PNG, JPG, JPEG, TIFF).

## Getting Started

### Prerequisites
- Google Colab environment (recommended for GPU access)
- Google Drive mounted to `/content/drive` for data and model storage.
- Python 3.8+
- `pip` package manager

### Installation

To set up your environment, run the following commands in your Colab notebook or terminal:

```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy scikit-image pillow tqdm opencv-python imquality piq
```

### Data Setup

Ensure your training data is organized in directories specified in the `CONFIG` dictionary. The `LOLDataset` expects paired low/high resolution images in separate directories, or a single directory for unpaired images with specific transformations.

Example data directory structure for paired images:

```
/content/drive/MyDrive/IPD/DATA/Train data/
├── lol_dataset/
│   ├── eval15/
│   │   ├── low/
│   │   └── high/
├── isro_data/
```

Update the `CONFIG` dictionary in the main script (`optifusionnet_res_multistage.py`) to point to your data directories:

```python
CONFIG = {
    "low_dir": "/content/drive/MyDrive/IPD/DATA/Train data/lol_dataset/eval15/low",
    "high_dir": "/content/drive/MyDrive/IPD/DATA/Train data/lol_dataset/eval15/high",
    # ... other configurations
}
```

For the modified `LOLDataset` that takes a single source directory, the config would be:

```python
CONFIG = {
    "image_source_dir": "/content/drive/MyDrive/IPD/isro_data",
    # ... other configurations
}
```

## Usage

### Training the Model

Run the `multistage_train()` function in the `optifusionnet_res_multistage.py` script. This will execute the entire multi-stage training pipeline, including meta-optimization and saving the best model checkpoint.

```python
if __name__ == "__main__":
    start = time.time()
    trained_model = multistage_train()
    print("Total elapsed:", time.time()-start)
```

### Running Inference

To enhance images using a trained model, use the `enhance_folder` function. Specify the path to your trained model, the input folder containing images to enhance, and an output folder for the enhanced images.

```python
MODEL_PATH = "optifusionnet_multistage_best.pth" # Path to your trained model
INPUT_FOLDER = "/content/drive/MyDrive/IPD/DATA/test/comtestSN" # Folder with images to enhance
OUTPUT_FOLDER = "/content/drive/MyDrive/IPD/opti_results" # Folder to save enhanced images

enhance_folder(MODEL_PATH, INPUT_FOLDER, OUTPUT_FOLDER)
```

### Visualizing Training Metrics

After training, you can plot the training and validation metrics (loss, PSNR, SSIM) using the provided plotting code.

```python
import matplotlib.pyplot as plt

# ... (assuming training_history and CONFIG are available from multistage_train output)

plt.figure(figsize=(18, 6))
# ... (plotting code from notebook)
plt.show()
```

### Evaluating Model with Quality Metrics

The notebook also includes code to evaluate the enhanced images using PSNR, SSIM, and BRISQUE metrics. This requires `scikit-image`, `imquality`, and `piq`.

```python
# ... (model loading and setup)

TEST_DIR = "/content/drive/MyDrive/IPD/DATA/test/comtestSN"
SAVE_DIR = "/content/drive/MyDrive/IPD/opti_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ... (test_dataset, test_loader setup)

# Inference loop with metric calculation and overlay on images
# ... (code from notebook's inference section)
```

## Configuration

The `CONFIG` dictionary at the beginning of the script controls various aspects of the training process, including:
- `low_dir`, `high_dir` (or `image_source_dir`): Dataset paths.
- `fast_size`, `hq_size`: Image resolutions for different training stages.
- `fast_adam_epochs`, `hq_adam_epochs`, `sgd_epochs`: Number of epochs for each training stage.
- `batch_size_fast`, `batch_size_hq`: Batch sizes.
- Meta-optimization parameters (`pso_iters`, `aco_iters`, etc.).
- `device`: 'cuda' or 'cpu'.
- `seed`: For reproducibility.
- `save_path`: Path to save the best model.