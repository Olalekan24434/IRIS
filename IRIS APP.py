import pandas as pd
import numpy as np
import joblib
import streamlit as st

model = joblib.load("rfiris.pkl")

st.title("IRIS FLOWER CLASSIFICATION APPLICATION")

st.write("Predict the species of an iris flower using a Random Forest Model")

form = st.form("iris form")

form.subheader("Enter Flower Measurement")

sepal_length = form.number_input(
    "sepal_length (cm)",
    min_value = 4.0,
    max_value = 8.0,
    value = 5.1
)

petal_length = form.number_input(
    "petal_length (cm)",
    min_value = 1.0,
    max_value = 7.0,
    value = 5.0
)

petal_width = form.number_input(
    "petal_width (cm)",
    min_value = 0.1,
    max_value = 2.5,
    value = 0.2
)

sepal_width= form.number_input(
    "sepal_width (cm)",
    min_value = 1.0,
    max_value = 4.5,
    value = 4.0
)

submit_button = form.form_submit_button("Predict")

if submit_button:
    input_data = pd.DataFrame({
        "sepal_length (cm)": [sepal_length],
        "petal_length (cm)" : [petal_length],
        "petal_width (cm)" : [petal_width],
        "sepal_width (cm)" : [sepal_width]
    })
    prediction = model.predict(input_data.values())

st.subheader("Prediction Result")

st.success(f"Predicted Species: {prediction[0]}")

