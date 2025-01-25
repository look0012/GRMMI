# Incorporating Graph Representation and Mutual Attention Mechanism for miRNA-mRNA Interaction Prediction

## Key Points
- This study introduces a **mutual attention mechanism**, enabling the model to effectively capture the complex interaction features between miRNAs and mRNAs, thereby uncovering additional latent associative information.
- The **mRNA sequence** is inputted in reverse order into the model, taking into account the biological characteristics of miRNA and mRNA binding.
- An improved **FastText** method is used as the pre-training model for RNA sequences, allowing for the generation of feature embeddings more aligned with the experimental objectives during deep mining.

## Overview

This repository contains the code for **GRMMI** , a deep learning model designed to predict interactions between microRNAs (miRNAs) and messenger RNAs (mRNAs). Understanding these interactions is critical for studying gene expression regulation and their implications in diseases. 

GRMMI overcomes limitations in existing models by efficiently handling RNA sequence complexities and graph structural information. It integrates both sequence-based and graph-based features to achieve high prediction performance.

### Key Features:
- **Pretraining of RNA Sequences**: GRMMI uses **FastText** for pretraining RNA sequences, which enriches feature extraction by embedding semantic and contextual information.
- **Graph Embedding**: The model incorporates **GraRep** graph embedding to capture topological features and node features from RNA interaction networks.
- **Multi-Level Feature Extraction**: Combines **CNN-BiLSTM** with a **mutual attention mechanism** to explore both local and global dependencies within RNA sequences.
- **MiRNA-mRNA Orientation**: Processes mRNA sequences by reversing their orientation to simulate complementary binding relationships with miRNAs.

The model achieves excellent results on the **MTIS-9214 dataset**, demonstrating strong predictive performance with an AUC of 0.9347 and an accuracy of 86.65%.

## Requirements

- Python 3.x
- TensorFlow >= 2.0
- Keras >= 2.0
- NumPy
- scikit-learn
- Matplotlib

You can install the required dependencies using `pip`:

```bash
pip install tensorflow keras numpy scikit-learn matplotlib
