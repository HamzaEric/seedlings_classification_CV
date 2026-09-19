# Plant Seedlings Classification - Computer Vision

This repository contains the computer vision pipeline for the Kaggle **Plant Seedlings Classification** competition. The objective is to accurately categorize 12 distinct species of plant seedlings using PyTorch-based Convolutional Neural Networks (CNNs) and transfer learning methodologies.

## Project Workflow & Notebooks

The core machine learning pipeline is housed within the `seedlings_classification_CV/Notebooks/` directory[cite: 2]. The project is structured into the following notebooks:

*   **`Preprocessing_&_data_prep.ipynb`**[cite: 2]: Handles Kaggle API authentication, dataset downloading, and directory formatting. It includes the implementation of a custom PyTorch `Dataset` to process the unclassified flat test directory, alongside stratified 33% subset splitting to create balanced training and validation data loaders.
*   **`Data Augmentation.ipynb`**[cite: 2]: Details the PyTorch `transforms` pipeline. This includes standardizing the dataset by resizing inputs to 224x224 pixels, converting them to tensors, and applying ImageNet mean/standard deviation normalization to optimize convergence.
*   **`Custom_CNN.ipynb`**[cite: 2]: Covers the architectural design, custom training loop, and baseline evaluation for a bespoke Convolutional Neural Network built and trained entirely from scratch.
*   **`Resnet_18_Transfer_Learning.ipynb`**[cite: 2]: Implements an advanced transfer learning approach utilizing a pre-trained ResNet-18 model. This notebook details freezing the feature extraction backbone, adapting the fully connected layer for 12 target classes, running the optimization scheduler, generating confusion matrices/classification reports, and formatting the final `submission.csv` for Kaggle scoring.

## Tech Stack

*   **Framework:** PyTorch (`torch`, `torchvision`, `torchinfo`)
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Metrics:** Scikit-learn (`accuracy_score`, `confusion_matrix`, `classification_report`)

## Setup & Installation

1. Clone the repository and navigate to the project root.
2. Ensure you have your `kaggle.json` API token configured locally (`~/.kaggle/kaggle.json`) or in your notebook environment.
3. You must actively accept the competition rules on Kaggle prior to downloading the dataset.
4. Install the required dependencies:
   ```bash
   pip install torch torchvision torchinfo pandas numpy matplotlib seaborn scikit-learn tqdm
