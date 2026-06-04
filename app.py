import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

st.title("Spam Message Detector")
st.write("Enter a message and the model will predict whether it is spam or ham.")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]

        ham_probability = probabilities[0]
        spam_probability = probabilities[1]

        st.subheader("Prediction")

        if prediction == "spam":
            st.error("Spam")
        else:
            st.success("Ham")

        st.write(f"Ham probability: {ham_probability:.4f}")
        st.write(f"Spam probability: {spam_probability:.4f}")