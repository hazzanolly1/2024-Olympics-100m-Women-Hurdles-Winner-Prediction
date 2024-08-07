
## Table of contents
1. Introduction
2. Dataset
3. Installation
4. Usage
5. Code Explanation
6. Class Initialization
7. Data Preprocessing  
8. Exploratory Data Analysis
9. Feature Engineering
10. Model Training
11. Winner Prediction
12. Results
13. Contributions
14. License(s)
## 1.0 Introduction
Olympics 2024 100m Women's Hurdles Winner Prediction:
This project aims to predict the winner of the 2024 Olympics 100m Women's Hurdles event.It uses aggregated historical performance data of Olympics 1988-2021, World Athletics Champpionship 2005-2023 and Diamond League 2015-2024 to train a machine learning model capable of predicting the finishing times for 8 finalist athletes and equally predict the Gold, Silver and Bronze medal winners with their anticipated finishing times.
## 2.0 Dataset
The dataset "C:/Users/LENOVO/Desktop/AI +Data Analytics Certifications Learning/AI Projects/2024 Olympics 100m Women Hurdles Winner Prediction/Olympics_2024_100m_Women_Hurdles.csv" used in this program contains historical performance data of athletes, including personal best times, season best times and other relevant features. The dataset is a CSV file with the following essential columns:
- Athlete: Name of the athlete
- Country: Country of the athlete
- Personal Best(s): Athlete's personal best time in seconds
- 2024 Season Best(s): Athlete's best time in the 2024 season in seconds
- 2024 Olympics Participant: Binary indicator of whether the athlete is participating in the 2024 Olympics


## 6.0 Installation
To install and run the 2024 Olympics 100 Women Hurdles Winner Prediction program, follow these steps:
1. Ensure you have the following dependencies installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

2. You can install these dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn




    
## 4.0 Usage
1. git clone https://github.com/hazzanolly1/2024-Olympics-100m-Women-Hurdles-Winner-Prediction.git
3. Navigate to the 2024 Olympics 100m Women Hurdles Winner Prediction directory:
cd 2024 Olympics 100 Women Hurdles Winner Prediction V1
4. Ensure you have Python 3.x installed. If not, download and install it from the official Python website.
5. Run the Python script:
Execute the Python script 2024 Olympics 100 Women Hurdles Winner Prediction V1.py  containing the code for loading, data preprocessing, EDA, Feature Engineering,Model training, and winner prediction on the 100m Women Hurdles dataset.
6. Adjust the instructions as needed based on your actual implementation and requirements.
Run the Program:


7. View Results:
The program will print the Root Mean Squared Error (RMSE) for each regression model, indicating the accuracy of the predictions.
Execute the Python script containing the provided code. 

1. Data Preparation
a) Loading the Dataset:
Abalone dataset was loaded into the Python environment from "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data".
b) Data Cleaning (Handling missing values):
A check for missing values was initially done. and subsequent filling missing values with the mean or median or removal of rows or columns with missing values in the dataset. 
c) Feature Engineering (Encoding Categorical Variable):The feature engineering process involves using One-hot encoding to convert the categorical variable "Sex" into a numerical format.The "Sex" column in the dataset is a categorical variable, meaning it contains non-numeric values.This involves creating dummy variables for each category of the "Sex" column.
The pd.get_dummies function from the Pandas library is used for subject one-hot encoding.This ensures that the resulting dataset consists of features (X) and the target variable (y) which are numerical input data suitable for training and evaluating machine learning models as most machine learning algorithms require numerical inputs. 
The drop_first=True parameter is set to drop the first dummy variable to avoid multi-collinearity.

2. Data Training:
a) Train-Test Split: The dataset was split into training and testing sets using the train_test_split function from sklearn.model_selection.                             This ensures that the model's performance can be evaluated on unseen data.
b) Model Training: Several regression algorithms were trained using the training data:
Linear Regression
Lasso Regression
Ridge Regression
Bagging Regression
Random Forest
LightGBM
c) Model Fitting: Each regression model was fitted to the training data using the fit method provided by the respective model classes. This involves learning the patterns and relationships between the input features and the target variable to predicting the number of rings in the abalone dataset which reflects the age.

3. Model Evaluation:
a) Predictions: After training the models, predictions were made on the test data using the predict method.
b) Evaluation Metrics: The performance of each model was evaluated using the root mean squared error (RMSE), a commonly used metric for regression tasks. The RMSE measures the average deviation of the predicted values from the actual values.
c) Comparison: The RMSE scores of all models were printed to compare their performance. This helps in selecting the best-performing model for the given dataset and task.
## 5.0 Code Explanation


## 6.0 Class Initialization
Class Initialization:
The OlympicsPrediction class is initialized with the file path("C:/Users/LENOVO/Desktop/AI +Data Analytics Certifications Learning/AI Projects/2024 Olympics 100m Women Hurdles Winner Prediction/Olympics_2024_100m_Women_Hurdles.csv") of the dataset. It attempts to read the file with different encodings to handle potential encoding issues.
## 7.0 Data Preprocessing
1. Handling Missing Values: Missing Numeric columns values in the dataset are filled with the mean values in its related column while categorical columns are also filled with the mode values in their related column.
2. Label Encoding: Categorical variables such as Athlete and Country are label-encoded for several reasons:
1. Machine Learning Model Requirements:
Many machine learning algorithms, especially those based on numerical calculations (like linear regression, random forests, and support vector machines), require input data to be numerical because they cannot process categorical data directly. Label encoding converts the categorical data into numerical values, making it suitable for these algorithms.
2. Maintaining Ordinal Information:
While label encoding assigns unique integers to each category, it does not imply any ordinal relationship between the categories. However, for algorithms that can work with categorical variables as numerical values, such as tree-based models (e.g., Random Forests), label encoding can be appropriate as these models can handle categorical data by splitting on the encoded labels.
3. Efficiency:Label encoding is more memory-efficient than one-hot encoding for categorical variables with a large number of categories. One-hot encoding can result in a very large number of binary features, especially if the categorical variable has many unique values. Label encoding keeps the feature space manageable.
4. Model Interpretation:Although label encoding introduces a sort of arbitrary numerical ordering, it can still be useful in models that do not treat these numbers as ordinal (like decision trees). In these models, the encoded labels are just identifiers that the model can use to split and make decisions.
However, it is essential to note that label encoding can introduce problems with algorithms that interpret numerical values as having an order or distance, which is not present in the original categorical data. For example, in linear regression or K-nearest neighbors, label encoding might not be suitable, and one-hot encoding or other encoding schemes might be preferred.

In the context of prediction winner code for subject 2024 Olympics 100m Women's Hurdles project, here are key reasons for using label encoding:
1. Athlete: Each athlete is uniquely identified. Label encoding provides a numerical identifier for each athlete, allowing the model to differentiate between them.
2. Country: Each country is also uniquely identified. Label encoding allows the model to incorporate country information as a feature without treating it as having any ordinal relationship.

In summary, label encoding is a straightforward and effective way to convert categorical variables into numerical form, suitable for many machine learning algorithms.



## 8.0 Exploratory Data Analysis
The Exploratory Data Analysis performs the following tasks:
1. Basic Statistics: Displays descriptive statistics of the dataset.
2. Identifies and converts the date columns into datetime data type to ensure A) That these date columns are interpreted correctly by the data processing tools and libraries. This prevents issues that could arise from treating date values as strings.B) Sorting and Indexing: Also,having date columns in datetime format allows for effective sorting, indexing, and time series analysis. This can be crucial for tasks like trend analysis and forecasting.
C) Data Integrity and Consistency:Ensuring that date columns are in the correct format helps maintain data integrity and consistency. This prevents issues related to incorrect date parsing or interpretation, which can lead to errors in downstream processing and analysis.
3. Correlation Heatmap: Shows the folowing:
A) Correlation/relationship coefficients between numeric pairs of variable features in the dataset.This helps in understanding how different features are related to each other.
B) Detects Multicollinearity:
Multicollinearity occurs when two or more independent variables are highly correlated. This can cause problems in regression models, leading to unreliable coefficient estimates. A correlation heatmap helps identify pairs of features that are highly correlated, indicating potential multicollinearity issues.
C) Feature Selection: By examining the correlations, we can identify which features are strongly related to the target variable and which are not. This helps in selecting the most relevant features for the model, potentially improving its performance.
D) Data Cleaning: A correlation heatmap can reveal unexpected correlations that might indicate data quality issues. For example, if two variables that should be independent show a strong correlation, this might point to an error in data collection or entry.
E) Insight into Data Structure:
Understanding the structure of the data and the relationships between variables can provide valuable insights that guide feature engineering and model selection.
4. Distribution Plots: Visualizes the distribution of personal best times and season best times.
5. Scatter Plots: Plots personal best and season best times for each athlete.

## 9.0 Feature Engineering 
1. Feature Selection: Selects features including personal best times, season best times, athlete, and country.
2. Target Variable: The target variable for subject regression model is the personal best time.

## 10.0 Model Split,Selection  and Training
1. Model Split: The train_regression_model method splits the data into training and testing sets and trains multiple regression models (RandomForest, LinearRegression, SVR) using MultiOutputRegressor. 
2. Model Selection: It selects the best model based on mean squared error evaluation metric.
3. Model Training: Trains the best model RandomForestRegressor and evaluates its performance using mean squared error.


## 11.0 Winner Prediction
The predict_winner method does the following:
1. Filters the data of athletes participating in the 2024 Olympics., ensuring the same feature engineering process, and predicts finishing times. 
2. It ensures the predicted times are not equal to either the personal best or season best times. 
3. Selects the best prediction based on criteria and maps back the encoded values to original labels.
4. Predicting Finalists: Predicts the 8 best athletes that will make the finals
5. Predict Finishing Times: Predicts the finishing times for the top 8 unique finalists. 
6. Medal Winners: Determines the Gold, Silver, and Bronze medal winners based on the predicted finishing times.

## 12.0 Results
After running the code, the predicted winner for the 2024 Olympics 100m Women's Hurdles is displayed with anticipated personal best and season best times.
Results
The predicted finishing times for the top 8 finalists and the medal winners are displayed, ensuring no duplicate athletes are in the final list.

## 13.0 Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request with any improvements or new features.
Please adhere to this project's `code of conduct`.


## 14.0 License
This project is licensed under the MIT License.