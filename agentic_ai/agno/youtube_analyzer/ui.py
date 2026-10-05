import streamlit as st
from youtube_analyzer import get_youtube_analyzer

@st.cache_resource
def get_agent():
    return get_youtube_analyzer()
agent = get_agent()

st.set_page_config(
    page_title="Youtube Analyzer",
    layout="centered"
)

st.title("▶️ AI Youtube Video Analyzer")

video_url = st.text_input("Enter the URL of youtube video: ")

instruction = st.text_input("Any extra instructions: ")

button = st.button("Analyze the video")

if video_url and button:
    with st.spinner("Analyzing video..."):
        response = agent.run(
            f"""
            Analyze this youtube video following the instruction (if any):
            Instruction: {instruction}
            URL: {video_url}
            """
        )
    st.markdown(response.content)





