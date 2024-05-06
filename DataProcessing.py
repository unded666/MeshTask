import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split


def extract_data(data_df: pd.DataFrame) -> np.ndarray:
    """
    extracts the data from the dataframe as a numpy array

    :param data_df: input dataframe, all cells being numeric
    :return: ndarray of shuffled data
    """

    data = data_df.to_numpy()

    return data

def scale_data (data: np.ndarray) -> tuple[np.ndarray, sklearn.preprocessing.MinMaxScaler]:
    """
    Scales the data using MinMaxScaler

    :param data: input data
    :return: tuple of scaled data and the scaler object
    """
    scaler = sklearn.preprocessing.MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler

def split_data(data: np.ndarray, train_size: float = 0.7, validate_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training, validation, and testing sets.

    :param data: input data
    :param train_size: proportion of the data to be used for training
    :param validate_size: proportion of the data to be used for validation
    :param test_size: proportion of the data to be used for testing
    :param seed: random seed used for replicability
    :return: tuple of training, validation, and testing sets
    """

    # First, split the data into training and remaining data
    train_data, remaining_data = train_test_split(data, train_size=train_size, random_state=seed)

    # Calculate the proportion of validation and test data from the remaining data
    remaining_proportion = validate_size / (validate_size + test_size)

    # Then, split the remaining data into validation and testing data
    validate_data, test_data = train_test_split(remaining_data, train_size=remaining_proportion, random_state=seed)

    return train_data, validate_data, test_data

def create_input_target(data: np.ndarray, target_index: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the data into input features and target feature. The last three columns are reserved for
    target features, and so any that are not listed in the target index are discarded.

    :param data: input data
    :param target_index: index of the target feature
    :return: tuple of input features and target feature
    """

    input_data = data[:, :-3]
    target_data = data[:, target_index]

    return input_data, target_data


if __name__ == '__main__':
    """
    Debugging code here
    """
    from FeatureEngineering import (one_hot_encode_feature,
                                    move_features_to_end,
                                    ordinal_encode_feature)

    # Base processing
    data = pd.read_csv('Data/study_performance.csv')
    data = data.drop(columns=['race_ethnicity', 'test_preparation_course'])
    data = one_hot_encode_feature(data, feature='gender')
    education_dict = {'some high school': 0,
                      'high school': 1,
                      "associate's degree": 2,
                      'some college': 3,
                      "bachelor's degree": 4,
                      "master's degree": 5}
    data = ordinal_encode_feature(data, 'parental_level_of_education', education_dict)
    lunch_values = data.lunch.unique()
    lunch_dict = {lunch_values[1]: 0,
                  lunch_values[0]: 1}
    data = ordinal_encode_feature(data, 'lunch', lunch_dict)
    data_reordered = move_features_to_end(data, ['math_score', 'reading_score', 'writing_score'])

    # debugging code
    data_np = extract_and_shuffle(data_reordered)
    data_scaled, scaler = scale_data(data_np)
    train_data, validate_data, test_data = split_data(data_scaled)
    train_input, train_target = create_input_target(train_data)





