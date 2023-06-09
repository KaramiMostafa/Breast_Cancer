# Efficient Feature Selection Using for Breast Cancer Dataset using Nested Cross-Validation

This repository hosts the source code for our study on feature selection in breast cancer prediction. We implemented multiple binary classification algorithms, and applied a novel approach using XGBoost for feature selection. The objective was to enhance model robustness, performance, and interpretability, while limiting overfitting. The overarching goal of our study was to offer a reliable machine learning approach for accurate breast cancer prediction, underscoring the significance of feature selection in healthcare prediction models.

## Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Datasets](#datasets)
5. [Author](#author)

## Introduction

In this study, we investigate feature selection in medical datasets for breast cancer prediction employing six binary classification algorithms. We consider two feature selection methods in connection with the eXtreme Gradient Boosting (XGBoost) classifier: nested cross-validation with grid search for hyperparameter tuning and K-fold cross-validation for evaluating feature importance. These techniques increase model robustness and interpretability while reducing overfitting. Our comparative study demonstrates that hyperparameter optimization enhances the performance of all tested algorithms. We identify the optimal feature count for maximizing model precision and propose a robust machine learning approach for precise breast cancer prediction. This underscores the pivotal role of feature selection in healthcare predictive models.

## Installation

The dataset used in this study is the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available and donated to the UCI Machine Learning Repository in 1995. It contains 569 instances, each representing a separate patient, and 32 features. These features offer a comprehensive basis for building a predictive model for breast cancer prognosis.

This version of the dataset is prior to any preprocessing steps. We encourage you to apply your own preprocessing techniques and explore different methodologies for the analysis.

If you wish to use this dataset for your work, please ensure you provide the appropriate citation. The citation for the Breast Cancer Wisconsin (Diagnostic) dataset is as follows:

```bash
@misc{misc_breast_cancer_wisconsin_(diagnostic)_17,
  author       = {Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.},
  title        = {{Breast Cancer Wisconsin (Diagnostic)}},
  year         = {1995},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5DW2B}
}
```
Please ensure the principles of fair usage and academic integrity by citing any materials or resources utilized in your research.

## Installation

1. Clone this repository to your local machine using the following command:
```bash
git clone [Repository URL]
```

2. Install the necessary Python packages via pip:
```bash
pip install -r requirements.txt
```

## Dependencies

Ensure you have the following Python libraries installed:

1. pandas
2. numpy
3. sklearn
4. matplotlib
5. seaborn
6. xgboost

## Contact
For questions or issues related to the code, please use the GitHub issue tracker. We appreciate your feedback!

## Author
1. Farzaneh Rastegari
2. Mostafa Karami
3. Sahand Hamzehei
4. Omid Akbarzadeh

