import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance Prediction App", layout="centered")

st.title("🏥 Insurance Prediction App")
st.write("Logistic Regression – Beginner Friendly")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance_data.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

st.write("✅ Columns found:", data.columns.tolist())

# -------------------------------
# Identify Target Column
# -------------------------------
target_column = data.columns[-1]   # LAST column as target

st.info(f"🎯 Target column used: **{target_column}**")

# -------------------------------
# Encode categorical data
# -------------------------------
le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])

# -------------------------------
# Split X and y
# -------------------------------
X = data.drop(target_column, axis=1)
y = data[target_column]

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# User Input
# -------------------------------
st.subheader("🧑 Enter Input Values")

input_data = []

for col in X.columns:
    value = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
    input_data.append(value)

input_data = np.array(input_data).reshape(1, -1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Prediction: YES")
    else:
        st.error("❌ Prediction: NO")
