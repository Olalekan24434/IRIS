import pandas as pd
import joblib
import streamlit as st

model = joblib.load("rfiris.pkl")

st.title("IRIS FLOWER CLASSIFICATION APPLICATION")

st.write("Predict the species of an iris flower using a Random Forest Model")

form = st.form("iris_form")

form.subheader("Enter Flower Measurement")

SepalLengthCm = form.number_input(
    "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1
)

SepalWidthCm = form.number_input(
    "Sepal Width (cm)", min_value=1.0, max_value=4.5, value=3.5
)

PetalLengthCm = form.number_input(
    "Petal Length (cm)", min_value=1.0, max_value=7.0, value=5.0
)

PetalWidthCm = form.number_input(
    "Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2
)

submit_button = form.form_submit_button("Predict")

if submit_button:

    input_data = pd.DataFrame({
        "sepal length (cm)": [SepalLengthCm],
        "sepal width (cm)": [SepalWidthCm],
        "petal length (cm)": [PetalLengthCm],
        "petal width (cm)": [PetalWidthCm]
    })

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    st.success(f"Predicted Species: {prediction[0]}")

