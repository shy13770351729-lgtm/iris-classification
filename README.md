# Iris Classification

A small reproducible machine-learning project for classifying Iris flowers into three species using logistic regression.

## Project purpose

This repository demonstrates a complete basic machine-learning workflow: loading data, exploring the dataset, splitting training and test sets, standardizing features, training a classifier, evaluating predictions, and visualizing results in Jupyter Notebook.

## Dataset

The project uses the Iris dataset included with `scikit-learn`. It contains 150 samples and four numerical features:

- sepal length
- sepal width
- petal length
- petal width

The target contains three classes: setosa, versicolor, and virginica. Because the dataset is bundled with scikit-learn, no manual data download is required.

## Project structure

```text
iris-classification/
├── README.md
├── analysis.ipynb
├── model.py
├── requirements.txt
└── .gitignore
```

`model.py` contains the reusable data-processing, training, and evaluation functions. `analysis.ipynb` demonstrates the experiment and displays the results.

## Installation

Clone the repository and install the dependencies:

```bash
git clone <your-repository-url>
cd iris-classification
pip install -r requirements.txt
```

## Reproduce the results

Run the Python script:

```bash
python model.py
```

Or start Jupyter Notebook:

```bash
jupyter notebook analysis.ipynb
```

Then run all cells from top to bottom.

## Reproducibility

The train/test split uses `random_state=42`, and stratified sampling is used to preserve the class distribution. The same code and dependency versions therefore produce reproducible experimental results.

## Method

1. Load the Iris dataset.
2. Inspect the data and class distribution.
3. Split the dataset into 80% training and 20% testing data.
4. Standardize the four input features.
5. Train a logistic regression model.
6. Predict the test samples.
7. Calculate accuracy, confusion matrix, and classification report.
8. Visualize the confusion matrix and feature distributions.

## Version

The first submitted version of this project is tagged as `v1.0`.

## Author

Course assignment project.
