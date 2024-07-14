import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\Datascience\Titanic-Dataset.csv")

# Display the first few rows
print(data.head())

# Get a summary of the dataset
print(data.info())

# Display statistical summary
print(data.describe())
