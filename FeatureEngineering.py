import sklearn
import pandas as pd

def ordinal_encode_feature(data_df: pd.DataFrame, feature: str, mapping: dict) -> pd.DataFrame:
    """
    This function ordinal encodes a given feature in a dataset. It replaces the values in the
    feature with the corresponding values in the mapping dictionary.

    Parameters:
    data_df (pd.DataFrame): The input dataset.
    feature (str): The feature to ordinal encode.
    mapping (dict): A dictionary mapping the original values to the new values.

    Returns:
    pd.DataFrame: The dataset with the ordinal encoded feature.
    """

    data_df[feature] = data_df[feature].map(mapping)

    return data_df

def one_hot_encode_feature(data_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function one-hot encodes a given feature in a dataset. It one_hot_encodes the
    desired feature and appends the new columns to the dataset as new_features with the
    suffix _{category}. The original feature is removed from the output dataset.

    Parameters:
    data_df (pd.DataFrame): The input dataset.
    feature (str): The feature to one-hot encode.

    Returns:
    pd.DataFrame: The dataset with the one-hot encoded feature.
    """

    one_hot_encoded = pd.get_dummies(data_df[feature], prefix=feature)
    data_df = data_df.drop(feature, axis=1)
    data_df = pd.concat([data_df, one_hot_encoded], axis=1)

    return data_df

def move_features_to_end(data_df: pd.DataFrame, feature_indices: list) -> pd.DataFrame:
    """
    This function moves the features specified by their indices to the right-hand side of the dataset.

    Parameters:
        data_df (pd.DataFrame): The input dataset.
        feature_indices (list): A list of indices of the features to move to the end of the dataset.

    Returns:
        pd.DataFrame: The dataset with the specified features moved to the end.
    """

    df_out = data_df.copy()
    early_cols = [col for col in data_df.columns if col not in feature_indices]
    late_cols = [col for col in data_df.columns if col in feature_indices]
    new_cols = early_cols + late_cols
    df_out = df_out[new_cols]

    return df_out

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('Data/study_performance.csv')
    # One-hot_encode the 'gender' feature
    data = one_hot_encode_feature(data, 'gender')
    data_reordered = move_features_to_end(data, ['math_score', 'reading_score', 'writing_score'])

