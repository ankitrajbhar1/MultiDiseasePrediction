import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Streamlit app
st.title("Breast Cancer Prediction")

# Sidebar for user input
st.sidebar.header("User Input")

# Display a sample of the dataset
if st.sidebar.checkbox("Show Sample Dataset"):
    st.write(X.head())

# User input features
st.sidebar.header("Enter Patient Information")

def user_input_features():
    features = {}
    for feature in X.columns:
        value = st.sidebar.slider(f"{feature}:", float(X[feature].min()), float(X[feature].max()))
        features[feature] = value
    return pd.DataFrame([features])

user_data_breast_cancer = user_input_features()

# Display user input
st.subheader("User Input for Breast Cancer:")
st.write(user_data_breast_cancer)

# Predict the disease for breast cancer
prediction_breast_cancer = rf_classifier.predict(user_data_breast_cancer)

# Display the prediction for breast cancer
st.subheader("Prediction for Breast Cancer:")
if prediction_breast_cancer[0] == 0:
    st.write("The model predicts that the patient is likely to have breast cancer.")
else:
    st.write("The model predicts that the patient is likely to be healthy.")

# Model accuracy
st.sidebar.subheader("Model Accuracy for Breast Cancer")
st.sidebar.write(f"Model Accuracy: {accuracy_score(y_test, rf_classifier.predict(X_test)):.2f}")
