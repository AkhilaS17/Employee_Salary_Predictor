import streamlit as st
import numpy as np
import joblib

# Load model

model = joblib.load('model/svm_salary_model.pkl')

# Page config
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# Custom CSS for dark theme and styled layout
st.markdown("""
    <style>
        body {
            background-color: #0f1117;
            color: white;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .reportview-container {
            background: #0f1117;
        }
        h1, h2, h3, h4 {
            color: white;
        }
        .stNumberInput > div {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        .stTextInput > div > div > input {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Gradient Header
st.markdown(
    """
    <div style='background: linear-gradient(to right, #00c6ff, #0072ff); 
                padding: 1.5rem; 
                border-radius: 20px; 
                margin-bottom: 2rem; 
                text-align: center; 
                color: white; 
                font-size: 40px; 
                font-weight: bold;'>
        Employee Salary Prediction Web App
    </div>
    """,
    unsafe_allow_html=True
)

# Layout columns
left, right = st.columns([2, 1])

with left:
    st.markdown(
        """
        <div style="background-color:#00c6ff; padding:1rem; border-radius:15px;">
            <h3>About the Project</h3>
            <p>This machine learning-based web application predicts whether an employee earns more than $50K per year based on five numeric features.</p>
            <p>The model is trained using a <b>Support Vector Classifier (SVC)</b> with feature scaling.</p>
            <p><b>Model Used:</b> Support Vector Classifier (SVC)</p>
            <p><b>Achieved Accuracy:</b> 84%</p>
            <h4>Input Field Details:</h4>
            <ul>
                <li><b>Age:</b> Employee’s age (e.g., 30)</li>
                <li><b>Education Number:</b> Numeric level of education (e.g., 10 = 10th grade)</li>
                <li><b>Capital Gain:</b> Profit made from capital assets</li>
                <li><b>Capital Loss:</b> Loss from capital assets</li>
                <li><b>Hours per Week:</b> Weekly work hours</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

with right:
    st.markdown("### Predict Employee Salary")

    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    education_num = st.number_input("Educational Number", min_value=1, max_value=20, value=10)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

    if st.button("Predict"):
        input_data = np.array([[age, education_num, capital_gain, capital_loss, hours_per_week]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            result = ">50K"
            color = "#1b5e20"  # dark green
        else:
            result = "<=50K"
            color = "#b71c1c"  # dark red

        st.markdown(f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 15px;
                margin-top: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            ">
                Predicted Salary: {result}
            </div>
        """, unsafe_allow_html=True)

