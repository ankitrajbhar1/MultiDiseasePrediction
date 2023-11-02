import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the heart disease dataset
heart_data = pd.read_csv('C:/Users/mpotb/OneDrive/Desktop/Multi Disease prediction/Dataset/heart.csv')

# Split the dataset into training and testing sets
X = heart_data.drop('target', axis=1)
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Streamlit app
st.title("Heart Disease Prediction")

# Sidebar for user input
st.sidebar.header("User Input")

# User input features
st.sidebar.header("Enter Patient Information")

def user_input_features():
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

    user_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    return user_data

user_data = user_input_features()

# Display user input
st.subheader("User Input:")
st.write(user_data)

# Predict the disease
prediction = rf_classifier.predict(user_data)

# Display the prediction
st.subheader("Prediction:")
if prediction[0] == 1:
    st.write("The model predicts that the patient is likely to have heart disease.")
else:
    st.write("The model predicts that the patient is likely to be healthy.")

# Model accuracy
st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"Model Accuracy: {accuracy_score(y_test, rf_classifier.predict(X_test)):.2f}")

if __name__ == '__main__':
    st.run()
