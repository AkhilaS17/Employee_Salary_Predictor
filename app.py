import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('model/svm_salary_model.pkl')

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("Employee Salary Prediction App")

st.markdown("Enter the details below to predict whether salary is >50K or <=50K.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)
education_num = st.number_input("Education Num", min_value=1, max_value=20, value=10)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

if st.button("Predict Salary"):
    try:
        # Prepare input
        input_df = pd.DataFrame([{
            'age': age,
            'educational-num': education_num,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = ">50K" if prediction == 1 else "<=50K"

        st.success(f"Predicted Salary: {result}")

    except Exception as e:
        st.error(f"Error occurred: {e}")
