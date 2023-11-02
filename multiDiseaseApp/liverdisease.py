import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a custom dataset for liver disease
# Replace 'custom_liver_data.csv' with the path to your dataset
# liver_data = pd.read_csv('C:\\Users\\mpotb\\OneDrive\\Desktop\\Multi Disease prediction\\Dataset\\indian_liver_patient.csv')
# liver_data = pd.read_csv('/path/to/your/dataset/custom_liver_data.csv')

# Load the liver disease dataset
liver_data = pd.read_csv('C:\\Users\\mpotb\\OneDrive\\Desktop\\Multi Disease prediction\\Dataset\\indian_liver_patient.csv')

# Define your input features (X) and target variable (y)
# Convert categorical variable 'Gender' into numerical values
liver_data['Gender'] = liver_data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Handle missing data by filling missing values with the mean
liver_data.fillna(liver_data.mean(), inplace=True)

# Separate input features (X) and target variable (y)
X = liver_data.drop('Dataset', axis=1)
y = liver_data['Dataset']



# # Define the features and target variable
# X = liver_data.drop('Liver_Disease', axis=1)
# y = liver_data['Liver_Disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Streamlit app
st.title("Liver Disease Prediction")

# Sidebar for user input
st.sidebar.header("User Input")

# User input features
st.sidebar.header("Enter Patient Information")

def user_input_features():
    features = {}
    for feature in X.columns:
        value = st.sidebar.slider(f"{feature}:", float(X[feature].min()), float(X[feature].max()))
        features[feature] = value
    return pd.DataFrame([features])

user_data_liver = user_input_features()

# Display user input
st.subheader("User Input for Liver Disease:")
st.write(user_data_liver)

# Predict the disease for liver disease
prediction_liver = rf_classifier.predict(user_data_liver)

# Display the prediction for liver disease
st.subheader("Prediction for Liver Disease:")
if prediction_liver[0] == 0:
    st.write("The model predicts that the patient is likely to have liver disease.")
else:
    st.write("The model predicts that the patient is likely to be healthy.")

# Model accuracy
st.sidebar.subheader("Model Accuracy for Liver Disease")
st.sidebar.write(f"Model Accuracy: {accuracy_score(y_test, rf_classifier.predict(X_test)):.2f}")
