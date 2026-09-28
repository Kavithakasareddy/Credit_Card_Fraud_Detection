import streamlit as st
import pandas as pd
from datetime import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check whether it is fraudulent.")

data = pd.read_csv("dataset/creditcard.csv")

st.success("Dataset loaded successfully!")
st.write("Total transactions:", len(data))

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

st.subheader("🔍 Enter Transaction Details")

selected_time = st.time_input(
    "🕐 Select Transaction Time",
    value=time(12, 0)
)

time_seconds = (
    selected_time.hour * 3600
    + selected_time.minute * 60
    + selected_time.second
)

st.write("Selected Time:", selected_time.strftime("%I:%M %p"))

amount = st.number_input(
    "💰 Transaction Amount",
    min_value=0.0,
    value=100.0
)

v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)

if st.button("🚨 Check Transaction"):

    input_data = [[
        time_seconds,
        v1,
        v2,
        v3,
        v4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
        amount
    ]]

    input_df = pd.DataFrame(
        input_data,
        columns=X.columns
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0]

    fraud_probability = probability[1] * 100

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
        st.write(
            "Fraud Probability:",
            round(fraud_probability, 2),
            "%"
        )
    else:
        st.success("✅ Transaction appears to be Normal.")
        st.write(
            "Fraud Probability:",
            round(fraud_probability, 2),
            "%"
        )
        