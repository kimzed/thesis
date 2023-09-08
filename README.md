# Mapping Agricultural Modernization in Kenya: Using Remote Sensing and Machine Learning

This repository contains the source code and datasets for a research project that leverages advanced remote sensing techniques to map greenhouses in Kenya.
## Repository Structure


```
thesis
│
├── analysis_and_testing       # Various scripts for data analysis and initial model testing
├── dataset.py                 # Functions and scripts to handle the dataset for training and evaluation
├── dataset_graphs.py          # Dataset manipulation for graph neural networks
├── evaluate.py                # Scripts for model evaluation and metrics computation
├── inference.py               # Model inference utilities
├── main.py                    # Main script for executing training, testing, and analysis workflows
├── MNIST                      # Proof of concept applied on the MNIST dataset
├── model                      # Contains various machine learning models used in the research
├── precision_recall_plot.py   # Script to plot precision-recall curve for model results
├── pre_processing             # Data pre-processing scripts and utilities
├── train.py                   # Scripts to handle model training
└── unit_test                  # Unit tests for various functionalities in the project
```

## Main Workflows

The main.py script contains various flags to control the type of analysis or model to run:

    DO_GRAPH_ANALYSIS: Execute graph-based model analysis
    DO_CNN_RUN: Run the Convolutional Neural Network models
    DO_RF_OBIA: Random Forest based Object-Based Image Analysis
    DO_RF: Basic Random Forest model training and evaluation

Toggle the flags to True or False as required.
