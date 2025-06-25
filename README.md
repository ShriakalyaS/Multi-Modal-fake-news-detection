#  Multiview-Fake-News-Detection

This project implements a robust fake news detection system using **multi-modal learning**, integrating four complementary views:
 Textual Content  
 Propagation Graphs  
 Stance Signals  
 Knowledge Graphs (KG)  

The system aims to overcome limitations of existing methods by leveraging diverse signals for early, accurate, and interpretable fake news detection.


## Key Features

- Separate preprocessing for each view
- Fusion of all views into a unified dataset
- MLP-based multi-view classifier with gated fusion
- Early stopping, learning rate scheduling
- Evaluation: Accuracy, Confusion Matrix, ROC & PR Curves

## Project Structure

project_root/
│
├── text_preprocessing.py # Text feature extraction
├── graph_preprocessing.py # Propagation graph feature extraction
├── stance_preprocessing.py # Stance signal feature extraction
├── kg_preprocessing.py # Knowledge graph feature extraction
│
├── fusion_dataset.py # Multi-view dataset for PyTorch
├── model.py # DETECTOR model with gated fusion
├── train.py # Training, validation & testing script
│
|__ datasets 
├── requirements.txt # Required Python packages
└── README.md # This file


##  How to Run

1. **Install Dependencies**

```bash
pip install -r requirements.txt

2.Prepare Data

Ensure your preprocessed feature dictionaries for each view (text, graph, stance, KG) are available.
Update the respective .py files with the correct logic to load these features.

3.Train & Evaluate

python train.py

##  Evaluation Metrics

Classification Report (Accuracy, Precision, Recall, F1)
Confusion Matrix
ROC Curve per Class
Precision-Recall Curve per Class

## Datasets Required

1.Twitter16 dataset for textual, stance, and propagation graph features
2.ConceptNet (filtered English subset) for knowledge graph features

## Keywords
Fake News Detection | Multi-Modal Learning | Stance Analysis | Knowledge Graphs | Propagation Graphs | Early Detection | Gated Fusion | Contrastive Learning

##  Acknowledgements

-ConceptNet for Knowledge Graph features
-Twitter16 dataset for textual, stance, and propagation graph data
-HuggingFace Transformers (if using BERT for text features)
-RGCN (Relational Graph Convolutional Networks) for inspiring KG feature modeling

