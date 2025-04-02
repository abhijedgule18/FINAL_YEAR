import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the tweet data."""
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Drop rows where 'Information Source' or 'Information Type' is 'Not labeled'
    data = data[(data['Information Source'] != 'Not labeled') & (data['Information Type'] != 'Not labeled')]

    # Remove rows that are completely empty
    data.dropna(how='all', inplace=True)

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Filter out rows that are marked as 'Not applicable' or 'Not related' in 'Informativeness'
    data = data[~data['Informativeness'].isin(['Not applicable', 'Not related'])]

    return data
