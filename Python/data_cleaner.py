import pandas as pd
import numpy as np


def clean_data(df, missing_threshold=0.5, outlier_threshold=3):
    """
    Clean the input DataFrame by handling missing values and outliers.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame to be cleaned.
    - missing_threshold: float (default=0.5)
        The threshold for the proportion of missing values in a column. Columns exceeding this threshold will be dropped.
    - outlier_threshold: float (default=3)
        The Z-score threshold for outlier detection. Rows with numeric values exceeding this threshold will be removed.

    Returns:
    - pandas.DataFrame
        The cleaned DataFrame.
    """
    # Copy the input DataFrame to avoid modifying the original data
    cleaned_df = df.copy()

    # Handling missing values
    missing_values = cleaned_df.isnull().sum() / len(cleaned_df)
    missing_columns = missing_values[missing_values > missing_threshold].index
    cleaned_df.drop(columns=missing_columns, inplace=True)

    # Handling outliers using Z-score
    num_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(
        (cleaned_df[num_columns] - cleaned_df[num_columns].mean()) / cleaned_df[num_columns].std())
    outlier_rows = z_scores > outlier_threshold
    cleaned_df = cleaned_df.mask(outlier_rows.any(axis=1)).dropna()

    return cleaned_df


# Example usage
# data = pd.read_csv('data.csv')
# cleaned_data = clean_data(data)

#Sample dataset with missing values and outliers
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, None, 8, 9],
    'C': [10, 11, 12, 13, 14],
    'D': [15, 16, 17, None, 19],
    'E': [20, 21, 22, 23, 24]
}
df = pd.DataFrame(data)

# Test the clean_data() function
cleaned_df = clean_data(df, missing_threshold=0.3, outlier_threshold=2)

# Print the original and cleaned DataFrames for comparison
print("Original DataFrame:")
print(df)
print("\nCleaned DataFrame:")
print(cleaned_df)
