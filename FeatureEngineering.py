import sklearn
import pandas as pd

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

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('Data/study_performance.csv')
    # One-hot_encode the 'gender' feature
    data = one_hot_encode_feature(data, 'gender')

