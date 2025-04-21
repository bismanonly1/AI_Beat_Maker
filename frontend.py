import streamlit as st
import requests

st.set_page_config(page_title = "AI Beat Generator")
st.title("AI Music Beat Maker")


st.markdown("Generate professional beats using AI. Select your genre and duration.")

genre = st.selectbox("Choose Genre", ["trap", "lofi", "drill", "house", "jazz", "afrobeat"])
duration = st.slider("Duration (seconds)", 5, 30, 10)

if st.button("Generate Beat"):
    with st.spinner("Generating beat..."):
        response = requests.post("http://localhost:8000/generate", data={"genre": genre, "duration": duration})
        with open("generated_beat.wav", "wb") as f:
            f.write(response.content)
        st.audio("generated_beat.wav")
        st.success("Your beat is ready!")