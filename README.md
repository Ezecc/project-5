# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project, provides an implementation of a machine learning model for identifying credit card customers that are most likely to churn. The project codebase includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package provide the flexibility of being run interactively or from the command-line interface (CLI).

The project data can be accessed from [here](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

The code uses Python 3.8.

## Files and data description

Overview of the files and data present in the project directory are as follows.

```
.
├── churn_library.py                            # A library of functions to find customers who are likely to churn
├── churn_notebook.ipynb                        # Contains the code provided that was refactored
├── churn_script_logging_and_tests.py           # Script that provides logging and testing functionalities
├── data                                        # Project data
│   └── bank_data.csv
├── Guide.ipynb                                 # Jupyter notebook provided for tips on getting started and troubleshooting
├── images                                      # Store images of EDA results
│   ├── eda
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transaction_distribution.png
│   └── results
│       ├── feature_importances.png
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs                                        # Store logs
│   └── churn_library.log
├── models                                      # Store trained models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── README.md                                   # Provide an overview of the project and instructions to use the code
├── requirements_py3.6.txt                      # Requirements file for running the project codes in python 3.6
└── requirements_py3.8.txt                      # Requirements file for running the project codes in python 3.8

```

## Running Files

How do you run your files? What should happen when you run your files?
The files can be exucuted as follows:

- Clone the project repository
- Create virtual environment (Python 3.6 or Python 3.8) for running the code. For instance:

  `$ conda create -n .venv python=3.8`

- Activate the virtual environment as follows:

  `$ conda activate .venv`

- Install the dependencies as follows:

  `$ pip install -r requirements_py3.8.txt`

- Run the Churn Library script as follows:

      ```$ ipython churn_library.```

  This performs the following tasks:

  - loads the data - perform categorical data encoding
  - Creates EDA figures and save them in the ./images/eda
  - Perform feature engineering on the provided data
  - Train the Random Forest and Logistic Regression models and save them in the ./models file. It also generates the roc curves which are saved in ./images/results/
  - perform feature importance for the random forest model and save the generated figures in ./images/results

- Run the churn_script_logging_and_tests.py script as follows:

  `$ ipython churn_script_logging_and_tests.py.`

  This perfoms the following tasks:

  - executes various tests on churn library functions and saves the logs in ./logs/churn_library.log
