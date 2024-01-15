import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import plotly.express as px

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

    education_levels = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:,.2f}")

        # Display additional visualizations with Plotly
        st.write(
            """
            ### Additional Visualizations
            """
        )

        # Bar chart showing average salary based on education level
        education_data = pd.DataFrame({"Education Level": education_levels})
        education_data["Average Salary"] = education_data["Education Level"].apply(
            lambda x: regressor.predict(np.array([[0, le_education.transform(x), 3]]))[0]
        )

        fig_bar = px.bar(
            education_data,
            x="Education Level",
            y="Average Salary",
            title="Average Salary Based on Education Level",
            labels={"Average Salary": "Average Salary ($)"},
            color="Average Salary",
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(fig_bar)

        # Line chart showing salary distribution across different countries
        countries_data = pd.DataFrame({"Country": countries})
        countries_data["Average Salary"] = countries_data["Country"].apply(
            lambda x: regressor.predict(np.array([[le_country.transform(x), 0, 3]]))[0]
        )

        fig_line = px.line(
            countries_data,
            x="Country",
            y="Average Salary",
            title="Salary Distribution Across Countries",
            labels={"Average Salary": "Average Salary ($)"},
            line_shape="linear",
            line_group="Country",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )

        st.plotly_chart(fig_line)
