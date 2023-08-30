import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv')  # Load the dataset from 'heart.csv' file
heartDisease = heartDisease.replace('?', np.nan)  # Replace '?' values with NaN

# Display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())  # Display the first few rows of the dataset

# Model Bayesian Network
model = BayesianModel([
    ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'), ('heartdisease', 'chol')
])  # Define the structure of the Bayesian Network

# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)  # Learn CPDs using Maximum Likelihood Estimator

# Inferencing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)  # Create an inference engine using VariableElimination

# Computing the Probability of HeartDisease given Age
print('\n1. Probability of HeartDisease given Age=28')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})  # Perform probabilistic inference
print(q['heartdisease'])  # Print the probability distribution

# Computing the Probability of HeartDisease given cholesterol
print('\n2. Probability of HeartDisease given cholesterol=100')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})  # Perform probabilistic inference
print(q['heartdisease'])  # Print the probability distribution
