import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
st.set_page_config(page_title="AI Disease Risk Predictor", layout="wide", page_icon="💊")
# Marquee
st.markdown("""
<marquee style='color: #FF4B4B; font-size:20px; font-weight:bold;' behavior="scroll" direction="left">
🚨 Stay Informed: Use this tool to check your Diabetes and Heart Disease risk early! 🚨
</marquee>
""", unsafe_allow_html=True)
# Title 
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Healthcare Risk & Insurance Cost Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>📊 Smart insights into Diabetes, Heart Disease & Insurance Cost — enter your data to begin!</p>", unsafe_allow_html=True)
st.write("")
#Sidebar
st.sidebar.image("logo1.png", use_column_width=True)
st.sidebar.title("📌 About the App")
st.sidebar.markdown("""
This app helps you:
- 🚑 Detect risk of **Diabetes** and **Heart Disease**
- 💰 Estimate **Health Insurance Cost**
- 📈 Explore health trend data
""")
st.sidebar.title("User Health Info")
name = st.sidebar.text_input("Name")
age_sidebar = st.sidebar.number_input("Age", 1, 120, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
st.sidebar.markdown("### BMI Calculator")
height_cm = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
bmi_status = "Not calculated"
bmi = 0.0
if height_cm > 0:
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    st.sidebar.markdown(f"#### Your BMI: `{bmi:.2f}`")
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_status = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
    st.sidebar.markdown(f"**Status:** {bmi_status}")
# Utility Functions 
def load_model(file_name):
    try:
        return pickle.load(open(file_name, 'rb'))
    except FileNotFoundError:
        st.error(f"Model file '{file_name}' not found.")
        return None
def show_insurance(bmi, age, condition):
    model_i = load_model('insurance_model.pkl')
    if model_i:
        sample = pd.DataFrame({
            'age': [age], 'sex': [1], 'bmi': [bmi],
            'children': [1], 'smoker': [1], 'region': [1]
        })
        cost = model_i.predict(sample)[0]
        st.markdown(f"### 💰 Insurance Cost Estimate for {condition} Risk: ₹{cost:,.2f}")
        st.info("💡 Tip: Maintain a healthy BMI and avoid smoking to reduce insurance costs.")
# Initialize prediction variables
diabetes_result = None
heart_result = None
insurance_result = None
# Tabs 
tab1, tab2, tab3 = st.tabs(["🦢 Diabetes Risk", "❤️ Heart Disease", "💵 Insurance Estimator"])
#Tab 1 
with tab1:
    st.subheader("🦢 Diabetes Risk Prediction")
    with st.expander("🔍 Fill in your health details:"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age_d = st.slider('Age', 20, 80, 30)
            gender_d = st.selectbox('Gender', ['Male', 'Female'])
            pregnancies = st.number_input('Pregnancies', 0)
        with col2:
            glucose = st.number_input('Glucose Level', 0)
            bp = st.number_input('Blood Pressure', 0)
            insulin = st.number_input('Insulin Level', 0.0)
        with col3:
            bmi_d = st.number_input('BMI', 0.0)
            pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
        gender_encoded = 1 if gender_d == 'Male' else 0
        diabetes_input = np.array([[pregnancies, glucose, bp, insulin, bmi_d, age_d, pedigree, gender_encoded]])
        if st.button("🚀 Predict Diabetes Risk"):
            model_d = load_model('diabetes_model.pkl')
            if model_d and glucose > 0 and bp > 0 and bmi_d > 0:
                prediction_d = model_d.predict(diabetes_input)
                if prediction_d[0] == 1:
                    diabetes_result = "High Risk"
                    st.error("⚠️ High Risk of Diabetes Detected")
                    show_insurance(bmi_d, age_d, "Diabetes")
                else:
                    diabetes_result = "Low Risk"
                    st.success("✅ You are at Low Risk for Diabetes")
            else:
                st.warning("🚫 Please enter valid positive values for Glucose, BP, and BMI.")
    st.markdown("---")
    df_d = pd.read_csv('diabetes.csv')
    df_d['AgeGroup'] = pd.cut(df_d['Age'], bins=[20,30,40,50,60,70,90], labels=['20s','30s','40s','50s','60s','70+'])
    age_trend_d = df_d.groupby('AgeGroup')['Outcome'].mean()
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(age_trend_d)
    with col2:
        fig1, ax1 = plt.subplots()
        sns.histplot(df_d['BMI'], bins=30, kde=True, ax=ax1, color="#00BFFF")
        st.pyplot(fig1)
# Tab 2 
with tab2:
    st.subheader("❤️ Heart Disease Risk Prediction")
    with st.expander("🔍 Enter your health stats:"):
        col1, col2 = st.columns(2)
        with col1:
            age_h = st.number_input("Age", 1, 100)
            trestbps = st.number_input("Resting Blood Pressure", 80, 200)
            chol = st.number_input("Cholesterol", 100, 600)
            thalach = st.number_input("Max Heart Rate", 60, 220)
        with col2:
            sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
            exang = st.selectbox("Exercise Angina (0 = No, 1 = Yes)", [0, 1])
            oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, step=0.1)
        input_h = np.array([[age_h, sex, cp, trestbps, chol, thalach, exang, oldpeak]])
        if st.button("🚀 Predict Heart Disease Risk"):
            model_h = load_model('heart_model.pkl')
            if model_h:
                prediction_h = model_h.predict(input_h)
                if prediction_h[0] == 1:
                    heart_result = "High Risk"
                    st.error("⚠️ Risk of Heart Disease Detected")
                    show_insurance(chol / 25, age_h, "Heart")
                else:
                    heart_result = "Low Risk"
                    st.success("✅ No Heart Disease Detected")
    st.markdown("---")
    df_h = pd.read_csv('heart.csv')
    df_h['AgeGroup'] = pd.cut(df_h['age'], bins=[20,30,40,50,60,70,100], labels=['20s','30s','40s','50s','60s','70+'])
    age_trend_h = df_h.groupby('AgeGroup')['target'].mean()
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(age_trend_h)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(df_h['chol'], bins=30, kde=True, ax=ax2, color="#FF4B4B")
        st.pyplot(fig2)
# Tab 3 
with tab3:
    st.subheader("💵 Insurance Cost Estimator")
    with st.expander("📋 Fill in your insurance info:"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age_i = st.number_input("Age", 18, 100)
            sex_i = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            bmi_i = st.number_input("BMI", 10.0, 50.0)
            children_i = st.number_input("Children", 0, 10, step=1)
        with col3:
            smoker_i = st.selectbox("Smoker", ["Yes", "No"])
            region_i = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
    sex_val = 1 if sex_i == "Male" else 0
    smoker_val = 1 if smoker_i == "Yes" else 0
    region_dict = {'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}
    region_val = region_dict[region_i]
    insurance_input = pd.DataFrame({
        'age': [age_i], 'sex': [sex_val], 'bmi': [bmi_i],
        'children': [children_i], 'smoker': [smoker_val], 'region': [region_val]
    })
    if st.button("📈 Predict Insurance Cost"):
        model_i = load_model('insurance_model.pkl')
        if model_i:
            predicted_cost = model_i.predict(insurance_input)[0]
            insurance_result = f"₹{predicted_cost:,.2f}"
            st.success(f"💰 Estimated Insurance Cost: {insurance_result}")
# Downloadable Report 
diabetes_result = diabetes_result or 'Not Available'
heart_result = heart_result or 'Not Available'
insurance_result = insurance_result or 'Not Available'
# diet plans
diabetes_diet = {
    "High Risk": """
    - 🥗 Eat more whole grains, legumes, green vegetables  
    - 🚫 Avoid sugary snacks, white bread, soda  
    - 💧 Stay hydrated and avoid fruit juices  
    - 🕒 Eat on time and maintain portion control
    """,
    "Low Risk": """
    - 🥦 Continue eating balanced meals  
    - 🍎 Include fruits, vegetables, and lean proteins  
    - 💧 Drink plenty of water  
    - 🏃 Stay physically active
    """,
    "Not Available": "⚠️ No prediction available. Please complete the input fields and try again."
}
heart_diet = {
    "High Risk": """
    - 🥬 Follow a DASH or Mediterranean diet  
    - ❌ Avoid salty, fried, and processed foods  
    - 🐟 Include omega-3-rich fish (like salmon)  
    - 🚶 Walk at least 30 minutes daily
    """,
    "Low Risk": """
    - 🥗 Maintain a heart-healthy diet  
    - 🍌 Eat potassium-rich foods (bananas, avocados)  
    - 🧂 Limit excessive salt and fat  
    - 🧘 Practice stress management
    """,
    "Not Available": "⚠️ No prediction available. Please complete the input fields and try again."
}
# Fetch diet text safely
diabetes_diet_text = diabetes_diet.get(diabetes_result, "⚠️ No diet recommendation available.")
heart_diet_text = heart_diet.get(heart_result, "⚠️ No diet recommendation available.")
#  report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9f9;
        color: #333;
        padding: 20px;
    }}
    .report-box {{
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: auto;
    }}
    h1 {{
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 30px;
    }}
    .section {{
        margin-bottom: 25px;
    }}
    .section-title {{
        font-size: 18px;
        font-weight: bold;
        color: #005B96;
        margin-bottom: 10px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 5px;
    }}
    .highlight {{
        background-color: #ffecec;
        padding: 12px;
        border-left: 5px solid #FF4B4B;
        border-radius: 5px;
    }}
    .success {{
        background-color: #e7f6e7;
        padding: 12px;
        border-left: 5px solid #2ecc71;
        border-radius: 5px;
    }}
    .neutral {{
        background-color: #f0f0f0;
        padding: 12px;
        border-left: 5px solid #999999;
        border-radius: 5px;
    }}
    .insurance-card {{
        background-color: #f0f8ff;
        padding: 20px;
        border: 2px dashed #00a8ff;
        border-radius: 10px;
        font-size: 18px;
        text-align: center;
        font-weight: bold;
        color: #0077b6;
    }}
    .diet-plan {{
        background-color: #fff8e1;
        border-left: 5px solid #f4b400;
        padding: 15px;
        border-radius: 8px;
        white-space: pre-line;
        font-size: 15px;
    }}
    .footer {{
        text-align: center;
        font-size: 12px;
        margin-top: 30px;
        color: #888;
    }}
</style>
</head>
<body>
<div class="report-box">
    <h1>🩺 Health Report</h1>

    <div class="section">
        <div class="section-title">📅 Date</div>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="section">
        <div class="section-title">🙍 User Information</div>
        <p><b>Name:</b> {name or 'User'}</p>
        <p><b>Age:</b> {age_sidebar}</p>
        <p><b>Gender:</b> {gender}</p>
        <p><b>BMI:</b> {bmi:.2f} ({bmi_status})</p>
    </div>
    <div class="section">
        <div class="section-title">🩺 Diabetes Risk</div>
        <div class="{ 'highlight' if diabetes_result == 'High Risk' else 'success' if diabetes_result == 'Low Risk' else 'neutral' }">
            {diabetes_result}
        </div>
    </div>
    <div class="section">
        <div class="section-title">❤️ Heart Disease Risk</div>
        <div class="{ 'highlight' if heart_result == 'High Risk' else 'success' if heart_result == 'Low Risk' else 'neutral' }">
            {heart_result}
        </div>
    </div>
    <div class="section">
        <div class="section-title">💰 Insurance Estimate</div>
        <div class="insurance-card">
            Estimated Cost: {insurance_result}
        </div>
    </div>
    <div class="section">
        <div class="section-title">🥦 Diet Plan for Diabetes</div>
        <div class="diet-plan">
            {diabetes_diet_text}
        </div>
    </div>
    <div class="section">
        <div class="section-title">🍎 Diet Plan for Heart Health</div>
        <div class="diet-plan">
            {heart_diet_text}
        </div>
    </div>
    <div class="footer">
        AI Disease Risk Predictor &copy; 2025 | Stay healthy 💖
    </div>
</div>
</body>
</html>
"""
# download button
st.download_button(
    label="📥 Download Complete HTML Report",
    data=html_report,
    file_name=f"health_report_{name.replace(' ', '_').lower() if name else 'user'}.html",
    mime="text/html"
)