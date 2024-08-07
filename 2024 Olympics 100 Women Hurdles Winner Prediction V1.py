#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class OlympicsPrediction:
    def __init__(self, file_path):
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Try an alternative encoding if utf-8 fails
        self.preprocessed_data = None
        self.regression_model = None
        self.label_encoders = {}

    def preprocess_data(self):
        # Handling missing values for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # Handling missing values for categorical columns
        categorical_cols = self.data.select_dtypes(include=[object]).columns
        self.data[categorical_cols] = self.data[categorical_cols].fillna(self.data[categorical_cols].mode().iloc[0])
        
        # Label encoding for categorical variables
        for column in ['Athlete', 'Country', 'Competition_Type']:  # Adjust column names based on your dataset
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le
        
        self.preprocessed_data = self.data
        return self.preprocessed_data

    def exploratory_data_analysis(self):
        # Basic statistics
        print(self.data.describe())
        
        # Identify and convert date columns if necessary
        date_columns = ['Date']  # Replace with your actual date columns
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
        
        # Correlation heatmap only for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.show()
        
        # Distribution of times
        sns.histplot(self.data['Personal Best(s)'], kde=True, color='blue', label='Personal Best(s)')
        sns.histplot(self.data['2024 Season Best(s)'], kde=True, color='red', label='2024 Season Best(s)')
        plt.legend()
        plt.show()
        
        # Plotting athletes' personal best times and 2024 season best times
        self.data['Athlete_Name'] = self.label_encoders['Athlete'].inverse_transform(self.data['Athlete'])
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.scatterplot(data=self.data, x='Athlete_Name', y='Personal Best(s)', color='blue', label='Personal Best(s)', ax=ax)
        sns.scatterplot(data=self.data, x='Athlete_Name', y='2024 Season Best(s)', color='red', label='2024 Season Best(s)', ax=ax)
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()
        
        # Creating a table for athletes sorted by personal best times
        sorted_athletes = self.data[self.data['2024 Olympics Participant'] == 1].sort_values(by='Personal Best(s)')
        table = sorted_athletes[['Athlete_Name', 'Personal Best(s)', '2024 Season Best(s)']].drop_duplicates(subset='Athlete_Name').reset_index(drop=True)
        print(table.head(40))  # Display all 40 participating athletes

    def feature_engineering(self):
        # Selecting features and target variable
        X = self.data[['Personal Best(s)', '2024 Season Best(s)', 'Athlete', 'Country']]
        y = self.data['Personal Best(s)']  # Use Personal Best(s) as target for regression model
        
        return X, y

    def train_regression_model(self, X, y):
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        models = {
            'RandomForest': RandomForestRegressor(),
        }
        
        best_model = None
        best_score = float('inf')
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = mean_squared_error(y_test, predictions)
            print(f"{model_name} Mean Squared Error: {score}")
            
            if score < best_score:
                best_score = score
                best_model = model
        
        self.regression_model = best_model
        print(f"Best Regression Model: {self.regression_model}")

    def predict_finalists(self):
        # Filter the data for 2024 participants
        participants_2024 = self.data[self.data['2024 Olympics Participant'] == 1]
        
        # Ensure the same feature engineering process
        X_2024 = participants_2024[['Personal Best(s)', '2024 Season Best(s)', 'Athlete', 'Country']]
        
        # Predict finishing times for the participants
        finishing_times = self.regression_model.predict(X_2024)
        
        # Add predictions to the dataframe and sort by predicted time
        participants_2024['Predicted_Finishing_Time'] = finishing_times
        finalists = participants_2024.sort_values(by='Predicted_Finishing_Time').drop_duplicates(subset='Athlete').head(8)
        
        # Print finishing times for the finalists
        for i, row in finalists.iterrows():
            athlete_name = self.label_encoders['Athlete'].inverse_transform([row['Athlete']])[0]
            country_name = self.label_encoders['Country'].inverse_transform([row['Country']])[0]
            print(f"Predicted Finishing Time for {athlete_name} ({country_name}): {row['Predicted_Finishing_Time']:.2f} seconds")
        
        # Determine Gold, Silver, and Bronze medal winners
        medal_winners = finalists.sort_values(by='Predicted_Finishing_Time').head(3)
        medals = ['Gold', 'Silver', 'Bronze']
        for medal, (i, row) in zip(medals, medal_winners.iterrows()):
            athlete_name = self.label_encoders['Athlete'].inverse_transform([row['Athlete']])[0]
            country_name = self.label_encoders['Country'].inverse_transform([row['Country']])[0]
            print(f"{medal} Medal: {athlete_name} from {country_name} with an expected finishing time of {row['Predicted_Finishing_Time']:.2f} seconds")

# Usage example
file_path = "C:/Users/LENOVO/Desktop/AI +Data Analytics Certifications Learning/AI Projects/2024 Olympics 100m Women Hurdles Winner Prediction/Olympics_2024_100m_Women_Hurdles.csv"
olympics_prediction = OlympicsPrediction(file_path)

# Preprocessing
preprocessed_data = olympics_prediction.preprocess_data()

# Exploratory Data Analysis
olympics_prediction.exploratory_data_analysis()

# Feature Engineering
X, y = olympics_prediction.feature_engineering()

# Train the regression model for finishing time prediction
olympics_prediction.train_regression_model(X, y)

# Predict the finalists and medal winners
olympics_prediction.predict_finalists()


# In[ ]:




