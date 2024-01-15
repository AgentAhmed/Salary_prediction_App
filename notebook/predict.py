import streamlit as st
import pickle
import numpy as np
import os

def load_model():
    # Get the absolute path to the 'saved_steps.pkl' file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'saved_steps.pkl')

    # Load the machine learning model and related data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Extract data from the loaded file
    regressor = data["model"]
    le_country = data["le_country"]
    le_education = data["le_education"]

    return regressor, le_country, le_education

# Load the model and related data
regressor, le_country, le_education = load_model()

def show_predict_page():
    st.title("Employee Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
