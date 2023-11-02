import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer


# Load the liver disease dataset
liver_data = pd.read_csv('C:/Users/mpotb/OneDrive/Desktop/Multi Disease prediction/Dataset/indian_liver_patient.csv')

# Convert 'Gender' to numerical values
liver_data['Gender'] = liver_data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
liver_data.fillna(liver_data.mean(), inplace=True)

# Separate input features (X) and target variable (y) for liver disease
X_liver = liver_data.drop('Dataset', axis=1)
y_liver = liver_data['Dataset']

# Train a Random Forest classifier for liver disease
rf_classifier_liver = RandomForestClassifier()
rf_classifier_liver.fit(X_liver, y_liver)

# Load the breast cancer dataset
data = load_breast_cancer()
X_breast_cancer = pd.DataFrame(data.data, columns=data.feature_names)
y_breast_cancer = pd.Series(data.target)

# Train a Random Forest classifier for breast cancer
rf_classifier_breast_cancer = RandomForestClassifier()
rf_classifier_breast_cancer.fit(X_breast_cancer, y_breast_cancer)

# Load the heart disease dataset
heart_data = pd.read_csv('C:/Users/mpotb/OneDrive/Desktop/Multi Disease prediction/Dataset/heart.csv')

# Separate input features (X) and target variable (y) for heart disease
X_heart = heart_data.drop('target', axis=1)
y_heart = heart_data['target']

# Train a Random Forest classifier for heart disease
rf_classifier_heart = RandomForestClassifier()
rf_classifier_heart.fit(X_heart, y_heart)

# Streamlit app
st.title("Multi Disease Prediction")

# Sidebar for user input
st.sidebar.header("User Input")

# User input features
st.sidebar.header("Enter Patient Information")

disease_selector = st.sidebar.radio("Select Disease:", ("Liver Disease", "Breast Cancer", "Heart Disease"))

if disease_selector == "Liver Disease":
    def user_input_features_liver():
        features = {}
        for feature in X_liver.columns:
            value = st.sidebar.slider(f"{feature}:", float(X_liver[feature].min()), float(X_liver[feature].max()))
            features[feature] = value
        return pd.DataFrame([features])

    user_data_liver = user_input_features_liver()

    # Display user input for liver disease
    st.subheader("User Input for Liver Disease:")
    st.write(user_data_liver)

    # Predict liver disease
    prediction_liver = rf_classifier_liver.predict(user_data_liver)

    # Display the prediction for liver disease
    # Predict the disease for liver disease
    prediction_liver = rf_classifier_liver.predict(user_data_liver)

    # Display the prediction for liver disease with custom styling
    st.subheader("Prediction for Liver Disease:")
    if prediction_liver[0] == 1:
        st.write('<p style="color: red; font-weight: bold;">The model predicts that the patient is likely to have liver disease.</p>', unsafe_allow_html=True)
    else:
        st.write('<p style="color: green; font-weight: bold;">The model predicts that the patient is likely to be healthy.</p>', unsafe_allow_html=True)


elif disease_selector == "Breast Cancer":
    def user_input_features_breast_cancer():
        features = {}
        for feature in X_breast_cancer.columns:
            value = st.sidebar.slider(f"{feature}:", float(X_breast_cancer[feature].min()), float(X_breast_cancer[feature].max()))
            features[feature] = value
        return pd.DataFrame([features])

    user_data_breast_cancer = user_input_features_breast_cancer()

    # Display user input for breast cancer
    st.subheader("User Input for Breast Cancer:")
    st.write(user_data_breast_cancer)

    # Predict breast cancer
    prediction_breast_cancer = rf_classifier_breast_cancer.predict(user_data_breast_cancer)

    # Display the prediction for breast cancer
    # Predict breast cancer
    prediction_breast_cancer = rf_classifier_breast_cancer.predict(user_data_breast_cancer)

    # Display the prediction for breast cancer with custom styling
    st.subheader("Prediction for Breast Cancer:")
    if prediction_breast_cancer[0] == 0:
        st.write('<p style="color: red; font-weight: bold;">The model predicts that the patient is likely to have breast cancer.</p>', unsafe_allow_html=True)
    else:
        st.write('<p style="color: green; font-weight: bold;">The model predicts that the patient is likely to be healthy.</p>', unsafe_allow_html=True)


elif disease_selector == "Heart Disease":
    def user_input_features_heart():
        features = {}
        # Define input fields based on your dataset columns
        age = st.sidebar.slider("Age", 29, 77)
        sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
        cp = st.sidebar.slider("Chest Pain Type (CP)", 0, 3)
        trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200)
        chol = st.sidebar.slider("Cholesterol (chol)", 126, 564)
        fbs = st.sidebar.selectbox("Fasting Blood Sugar (fbs)", ("< 120 mg/dl", "> 120 mg/dl"))
        restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", ("Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"))
        thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 71, 202)
        exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", ("No", "Yes"))
        oldpeak = st.sidebar.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2)
        slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (slope)", 0, 2)
        ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 4)
        thal = st.sidebar.selectbox("Thalassemia (thal)", ("Normal", "Fixed defect", "Reversible defect"))

        # Map sex to numerical value
        sex = 1 if sex == "Male" else 0

        # Map fbs and exang to numerical values
        fbs = 1 if fbs == "> 120 mg/dl" else 0
        exang = 1 if exang == "Yes" else 0

        # Map restecg and thal to numerical values
        restecg_mapping = {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        }

        thal_mapping = {
            "Normal": 1,
            "Fixed defect": 2,
            "Reversible defect": 3
        }

        restecg = restecg_mapping[restecg]
        thal = thal_mapping[thal]

        features["age"] = age
        features["sex"] = sex
        features["cp"] = cp
        features["trestbps"] = trestbps
        features["chol"] = chol
        features["fbs"] = fbs
        features["restecg"] = restecg
        features["thalach"] = thalach
        features["exang"] = exang
        features["oldpeak"] = oldpeak
        features["slope"] = slope
        features["ca"] = ca
        features["thal"] = thal

        return pd.DataFrame([features])

    user_data_heart = user_input_features_heart()

    # Display user input for heart disease
    st.subheader("User Input for Heart Disease:")
    st.write(user_data_heart)


   # Predict heart disease
    prediction_heart = rf_classifier_heart.predict(user_data_heart)

    # Display the updated prediction for heart disease with custom styling
    st.subheader("Prediction for Heart Disease:")
    if prediction_heart[0] == 1:
        st.write('<p style="color: red; font-weight: bold;">The model predicts that the patient is likely to have heart disease.</p>', unsafe_allow_html=True)
    else:
        st.write('<p style="color: green; font-weight: bold;">The model predicts that the patient is likely to be healthy.</p>', unsafe_allow_html=True)



# # Model accuracy
# st.sidebar.subheader("Model Accuracy")
# if disease_selector == "Liver Disease":
#     st.sidebar.write(f"Model Accuracy for Liver Disease: {accuracy_score(y_liver, rf_classifier_liver.predict(X_liver)):.2f}")
# elif disease_selector == "Breast Cancer":
#     st.sidebar.write(f"Model Accuracy for Breast Cancer: {accuracy_score(y_breast_cancer, rf_classifier_breast_cancer.predict(X_breast_cancer)):.2f}")
# elif disease_selector == "Heart Disease":
#     st.sidebar.write(f"Model Accuracy for Heart Disease: {accuracy_score(y_heart, rf_classifier_heart.predict(X_heart)):.2f}")

# Copyright
st.markdown("© 2023 by ShaliniBawankule, AnkitRajbhar, AvinashChandore")
