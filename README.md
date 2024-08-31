[![Codacy Badge](https://app.codacy.com/project/badge/Grade/624323824c504fbf82755743f47894fe)](https://app.codacy.com/gh/Abhinav330/ML-implementation-on-Classified-data/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/ML-implementation-on-Classified-data/matplotlib?color=beige)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/ML-implementation-on-Classified-data/numpy?color=red)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/ML-implementation-on-Classified-data/pandas?color=silver)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/ML-implementation-on-Classified-data/scikit-learn?color=silver)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/ML-implementation-on-Classified-data/scipy?color=beige)
![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/Abhinav330/ML-implementation-on-Classified-data/seaborn?color=gold)
![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/Abhinav330/ML-implementation-on-Classified-data?color=dark%20green)
![GitHub repo size](https://img.shields.io/github/repo-size/Abhinav330/ML-implementation-on-Classified-data)

# ML-implementation-on-Classified-data
This repository implements a K-Nearest Neighbors (KNN) model in Python for predicting loan defaults. It analyzes the KNN_Project_Data dataset (features &amp; binary target 'TARGET CLASS').

# Code Summary

This Python script demonstrates the use of the K-Nearest Neighbors (KNN) classification algorithm for a binary classification task. It imports various libraries, loads a dataset named 'KNN_Project_Data', scales the data, and trains a KNN model to predict the 'TARGET CLASS'. The code also performs model evaluation and hyperparameter tuning.

## Data Loading and Exploration

The code begins by importing necessary libraries, loading the dataset 'KNN_Project_Data' using pandas, and displaying dataset information using `df.info()`. The dataset is assumed to contain features and a binary target variable ('TARGET CLASS'). 

## Data Preprocessing

Data preprocessing steps include:
- Scaling the feature data using StandardScaler from scikit-learn to ensure that all features have the same scale.
- Splitting the dataset into training and testing sets using `train_test_split()`.

## Model Building

The code then proceeds to build a K-Nearest Neighbors (KNN) classifier model with the following steps:
- Importing KNeighborsClassifier from scikit-learn.
- Initializing a KNN model with a specified number of neighbors (in this case, `n_neighbors=1`).
- Fitting the model to the training data.
- Making predictions on the test data.

## Model Evaluation

The code evaluates the KNN model by:
- Importing classification_report, confusion_matrix, and accuracy_score from scikit-learn.
- Printing the confusion matrix and classification report, which includes metrics such as precision, recall, and F1-score.
- Performing hyperparameter tuning by iterating through different values of 'n_neighbors' and collecting error values. This helps to find an optimal 'n_neighbors' value.
- Plotting the error values against 'n_neighbors' to visualize the trade-off between bias and variance.
- Rebuilding the KNN model with the optimal 'n_neighbors' value found during hyperparameter tuning.
- Printing the final confusion matrix, classification report, and accuracy score.

The code aims to find the best 'n_neighbors' value that maximizes the model's performance.

