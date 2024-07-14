# Survival-Prediction
Created a simple prediction engine on survivial prediction aims to predict the survival of passengers on the Titanic based on various features using machine learning algorithms.

Predicting the survival of passengers on the Titanic has been a subject of study for many years. The objective of this project is to learn in-depth about the implementation of machine learning algorithms using Python, with the Titanic dataset providing a perfect opportunity to explore classification tasks.
The data for this project comes from the Kaggle Titanic dataset, which includes various features such as passenger age, gender, class, fare, and more. The target variable is binary, indicating whether a passenger survived or not.
The ultimate goal of this project is to build a robust prediction model that can accurately classify whether a passenger survived based on the available features.

**Features Used**
  • Pclass: Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
  • Sex: Gender
  • Age: Age of the passenger
  • SibSp: Number of siblings/spouses aboard the Titanic
  • Parch: Number of parents/children aboard the Titanic
  • Fare: Passenger fare
  • Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

**Machine Learning Models Used**
  To achieve the best possible prediction accuracy, the machine learning models and techniques used were
      **Random Forest Classifier**: An ensemble learning method for classification. 
      **GridSearchCV**: For hyperparameter tuning to find the best model parameters.

      
**Implementation**
  Tools and Libraries
    • **Python:** The programming language used for implementation
    • **PyCharm:** The IDE used for development
    • **Pandas:** For data manipulation and analysis
    • **NumPy:** For numerical operations
    • **Scikit-learn:** For building and evaluating machine learning models
    • **Matplotlib and Seaborn:** For data visualization


**Steps**
  **• Data Loading and Exploration:** Load the dataset and perform exploratory data analysis (EDA) to understand the data.
  **• Data Preprocessing:** Handle missing values, encode categorical features, and scale numerical features.
  **• Model Building:** Train multiple classification models and evaluate their performance.
  **• Model Evaluation:** Use metrics such as accuracy, precision, recall, and F1-score to evaluate model performance.
  **• Prediction:** Use the best-performing model to make predictions on the test dataset.
  
  
**Feature Selection**
  To identify the most important features for predicting survival, a Random Forest model was used. The top features selected were:
    • Fare
    • Sex_male
    • Age
    • Pclass
    • SibSp

    
**Results**
  The Random Forest Classifier achieved the highest accuracy of **82%.**
  The logistic regression model performed reasonably well with an accuracy of **80%.**

**Conclusion**
  This project successfully demonstrates the process of building and evaluating machine learning models for classification tasks using the Titanic dataset. The achieved accuracy indicates that the models can predict the survival of passengers with a high degree of confidence.
