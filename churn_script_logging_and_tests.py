'''
Script for tests on customer churn model code

Author: Chijioke Eze

Date: 11.12.2022

'''

# Import libaries
import os
import logging
from math import ceil
import churn_library

# Configure basic logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def test_import():
    '''
    Test import_data() function from the churn_library module
    '''
    # Check if data file is available
    try:
        dataframe = churn_library.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    # Test the dataframe
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info(
            'Rows: %d\tColumns: %d',
            dataframe.shape[0],
            dataframe.shape[1])
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    Test perform_eda() function from the churn_library module
    '''
    dataframe = churn_library.import_data("./data/bank_data.csv")
    try:
        churn_library.perform_eda(dataframe=dataframe)
        logging.info("Testing perform_eda function: SUCCESS")
    except KeyError as err:

        logging.error('Column "%s" not found', err.args[0])
        raise err

    # Check if `churn_distribution.png` is created
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info('File %s is available', 'churn_distribution.png')
    except AssertionError as err:
        logging.error('No such file found')
        raise err

    # Check if `customer_age_distribution.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/customer_age_distribution.png") is True
        logging.info('File %s is available', 'customer_age_distribution.png')
    except AssertionError as err:
        logging.error('No such file found')
        raise err

    # Check if `marital_status_distribution.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/marital_status_distribution.png") is True
        logging.info('File %s is available', 'marital_status_distribution.png')
    except AssertionError as err:
        logging.error('No such file found')
        raise err

    # Check if `total_transaction_distribution.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png") is True
        logging.info(
            'File %s is available',
            'total_transaction_distribution.png')
    except AssertionError as err:
        logging.error('No such file found')
        raise err

    # Assert if `heatmap.png` is created
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File %s is available', 'heatmap.png')
    except AssertionError as err:
        logging.error('Not such file found')
        raise err


def test_encoder_helper():
    '''
    Test encoder_helper() function from the churn_library module
    '''
    # Load DataFrame
    dataframe = churn_library.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Get categorical features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        encoded_df = churn_library.encoder_helper(
            dataframe=dataframe,
            category_lst=[],
            response=None)

        # Check if there is alterations in the data
        assert encoded_df.equals(dataframe) is True
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=[]): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=[]): ERROR")
        raise err

    try:
        encoded_df = churn_library.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns,
            response=None)

        # Check column names for correctness
        assert encoded_df.columns.equals(dataframe.columns) is True

        # Check if encoded df is different from the main dataframe
        assert encoded_df.equals(dataframe) is False
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): ERROR")
        raise err

    try:
        encoded_df = churn_library.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns,
            response='Churn')

        # Column names should be different
        assert encoded_df.columns.equals(dataframe.columns) is False

        # Data should be different
        assert encoded_df.equals(dataframe) is False

        # Check if number of columns in encoded_df is equal to the sum of
        # columns in data_frame and the newly created columns from cat_columns
        assert len(
            encoded_df.columns) == len(
            dataframe.columns) + len(cat_columns)
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): ERROR")
        raise err


def test_perform_feature_engineering():
    '''
    Test perform_feature_engineering() function in the churn_library file
    '''
    # Load the DataFrame
    dataframe = churn_library.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    try:
        (_, X_test, _, _) = churn_library.perform_feature_engineering(
            dataframe=dataframe,
            response='Churn')

        # Check if `Churn` column is present in `data_frame`
        assert 'Churn' in dataframe.columns
        logging.info(
            "Testing perform_feature_engineering. `Churn` column is present: SUCCESS")
    except KeyError as err:
        logging.error(
            'The `Churn` column is not present in the DataFrame: ERROR')
        raise err

    try:
        # X_test size should be 30% of `data_frame`
        assert (
            X_test.shape[0] == ceil(
                dataframe.shape[0] *
                0.3)) is True
        logging.info(
            'Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR')
        raise err


def test_train_models():
    '''
    Test train_models() function from the churn_library file
    '''
    # Load the DataFrame
    dataframe = churn_library.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Feature engineering
    (X_train, X_test, y_train, y_test) = churn_library.perform_feature_engineering(
        dataframe=dataframe, response='Churn')

    # Assert if `logistic_model.pkl` file is present
    try:
        churn_library.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('File %s is available', 'logistic_model.pkl')
    except AssertionError as err:
        logging.error('No such file')
        raise err

    # Assert if `rfc_model.pkl` file is present
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('File %s is available', 'rfc_model.pkl')
    except AssertionError as err:
        logging.error('No such file')
        raise err

    # Assert if `roc_curve_result.png` file is present
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('File %s is available', 'roc_curve_result.png')
    except AssertionError as err:
        logging.error('No such file')
        raise err

    # Assert if `rfc_results.png` file is present
    try:
        assert os.path.isfile('./images/results/rf_results.png') is True
        logging.info('File %s is available', 'rf_results.png')
    except AssertionError as err:
        logging.error('No such file')
        raise err

    # Assert if `logistic_results.png` file is present
    try:
        assert os.path.isfile('./images/results/logistic_results.png') is True
        logging.info('File %s is available', 'logistic_results.png')
    except AssertionError as err:
        logging.error('No such file')
        raise err

    # Assert if `feature_importances.png` file is present
    try:
        assert os.path.isfile(
            './images/results/feature_importances.png') is True
        logging.info('File %s is available', 'feature_importances.png')
    except AssertionError as err:
        logging.error('No such file')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
