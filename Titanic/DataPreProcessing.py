import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\Datascience\Titanic-Dataset.csv")

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = data.drop(columns=['Cabin', 'Name', 'Ticket'])

# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Separate features and target variable
X = data.drop(columns=['Survived'])
y = data['Survived']
print(data.info())