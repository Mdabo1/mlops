# MLOPS Homework 1

## Income Prediction Model
This repository contains a machine learning pipeline for predicting income levels based on demographic data. The pipeline includes data generation, preprocessing, model training, and evaluation.  

## Files
- data_creation.py: Generates synthetic data using the Faker library and splits it into training and testing sets.  
- model_preprocessing.py: Preprocesses the data by excluding certain columns, scaling numerical features, and encoding the target variable.  
- model_preparation.py: Trains a linear regression model using the preprocessed training data and saves the trained model.  
- model_testing.py: Evaluates the trained model on the testing data and calculates the mean squared error.  
- pipeline.sh: A bash script that runs the entire pipeline by executing the Python scripts in the correct order.  
## Dependencies

Clone the repository:  

```
git clone https://github.com/your-username/income-prediction-model.git  
```

Install the required dependencies:  
``` 
pip install pandas scikit-learn Faker joblib  
```
Run the pipeline:  
```
bash pipeline.sh
```
