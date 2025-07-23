import streamlit as st
import joblib
import pandas as pd

# Set Streamlit page config
st.set_page_config(page_title="Salary Predictor", layout="wide")

# ---------------- CSS Section ----------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: #e0f2f1;
        padding: 2rem;
    }
    .title-box {
        background: linear-gradient(to right, #0288d1, #26c6da);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .info-box {
        background: linear-gradient(to right, #00c6ff, #7ee8fa);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.15);
    }
    .predict-box {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.15);
    }
    .result-box {
        background-color: #004d40;
        color: white;
        padding: 10px;
        margin-top: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<div class='title-box'><h1>Employee Salary Prediction Web App</h1></div>", unsafe_allow_html=True)
st.write("")

# ---------------- Load the saved model ----------------
model = joblib.load('model/svm_salary_model.pkl')

# ---------------- Layout ----------------
left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.markdown("""
    <div class="info-box">
        <h3>About the Project</h3>
        <p>This machine learning-based web application predicts whether an employee earns more than $50K per year based on five numeric features. The model is trained using a Support Vector Classifier (SVC) with feature scaling.</p>
        <b>Model Used:</b> Support Vector Classifier (SVC) <br>
        <b>Achieved Accuracy:</b> 84%
        <h4>Input Field Details:</h4>
        <ul>
        <li><b>Age:</b> Employee’s age (e.g., 30)</li>
        <li><b>Education Number:</b> Numeric level of education (e.g., 10 = 10th grade)</li>
        <li><b>Capital Gain:</b> Profit made from capital assets</li>
        <li><b>Capital Loss:</b> Loss from capital assets</li>
        <li><b>Hours per Week:</b> Weekly work hours</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='predict-box'>", unsafe_allow_html=True)

    st.subheader("Predict Employee Salary")

    age = st.number_input("Age", 17, 90, 30)
    education_num = st.number_input("Educational Number", 1, 16, 10)
    capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
    hours_per_week = st.number_input("Hours per Week", 1, 100, 40)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([{
                'age': age,
                'educational-num': education_num,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week
            }])
            prediction = model.predict(input_df)[0]
            result = ">50K" if prediction == 1 else "<=50K"

            st.markdown(f"<div class='result-box'>Predicted Salary: {result}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error occurred: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
