# Multimodal Neurodegenerative Disease Research

A unified framework for early detection and progression prediction of Alzheimer's and Parkinson's diseases using deep learning and multimodal fusion.

## Overview
This project integrates structural MRI imaging with clinical and demographic metadata (Age, MMSE, CDR, UPDRS) to improve diagnostic sensitivity, particularly for prodromal stages of neurodegenerative diseases.

## Key Features
- **Multimodal Fusion**: Late fusion architecture combining CNN-based image features with MLP-based clinical features.
- **Disease Coverage**: Specialized branches for both Alzheimer’s Disease (AD) and Parkinson’s Disease (PD).
- **Explainability**: Integrated Grad-CAM visualization to interpret model focus regions in neuro-imaging.
- **Simulated Metadata**: Realistic clinical datasets generated to augment image-only cohorts for development.

## Project Structure
- `data/`: Raw and preprocessed imaging data (excluded from repo).
- `metadata/`: Clinical datasets and simulated records.
- `models/`: Modular architecture including `fusion_model.py`.
- `scripts/`: Implementation scripts for preprocessing, training, and evaluation.
- `results/`: Output metrics, confusion matrices, and Grad-CAM visualizations.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, OpenCV (for explainability)

### Training
To train the multimodal model (defaulting to Parkinson's):
```bash
python scripts/train_multimodal.py
```

### Evaluation
To evaluate the trained model:
```bash
python scripts/evaluate_multimodal.py
```

## Methodology
The framework uses a ResNet-18 backbone for MRI feature extraction and a Multi-Layer Perceptron (MLP) for clinical data. Features are concatenated in a fusion head for final classification.

## Future Work
- Integration of real-world Ghanaian patient metadata.
- Refinement of 3D MRI processing.
- Fairness auditing across diverse demographic subgroups.
