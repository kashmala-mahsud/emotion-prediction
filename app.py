# app.py
import streamlit as st
import joblib

# Load Logistic Regression model + TFIDF
model = joblib.load('logistic_model.pkl')  # 🔁 Changed model name
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_mapping = joblib.load('label_mapping.pkl')
reverse_mapping = {v: k for k, v in label_mapping.items()}



# Optional: Add emoji for emotions
emoji_map = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'love': '❤️',
    'fear': '😨',
    'surprise': '😲'
}

# Streamlit App
st.set_page_config(page_title="Emotion Predictor", page_icon="🔮", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #6C63FF;'>🎭 Emotion Detection App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter a sentence and let the AI detect the emotion!</p>",
    unsafe_allow_html=True
)

# Input
user_input = st.text_area("Enter your text here 👇", height=150)

# Predict Button
if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform input and predict
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        predicted_emotion = reverse_mapping[prediction]
        emoji = emoji_map.get(predicted_emotion, '')

        # Display result
        st.success(f"**Predicted Emotion:** `{predicted_emotion.upper()}` {emoji}")
