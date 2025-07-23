# Employee Salary Predictor 🧠💰

This is a Streamlit web app that predicts whether an employee earns more than $50K or not, based on input features like:

- Age
- Education Number
- Capital Gain
- Capital Loss
- Hours per Week

## 🔍 How it Works

This app uses a pre-trained Support Vector Machine (SVM) model built using Scikit-learn. It was trained on a cleaned version of the Adult Income dataset.

## 🚀 How to Use

1. Open the app [here](https://your-link.streamlit.app)
2. Enter the employee details in the form
3. Click **Predict Salary**
4. See the result!

## 🗂 Files in this Repo

- `app.py`: Streamlit app code
- `requirements.txt`: Python libraries needed
- `model/svm_salary_model.pkl`: Trained model

## 🛠 Built With

- Streamlit
- scikit-learn
- pandas
- joblib
