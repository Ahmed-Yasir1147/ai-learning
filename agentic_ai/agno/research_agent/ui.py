import streamlit as st
from research_agent_v2 import build_research_workflow

@st.cache_resource
def get_workflow():
    return build_research_workflow()
workflow = get_workflow()

st.set_page_config(
    page_title="Research Agent",
    layout="centered"
)

st.title("🔬 AI Research Agent")

problem_text = st.text_input("Type the problem or question you want to research about: ")

research_btn = st.button("Research")

if problem_text and research_btn:
    with st.spinner("Researching..."):
        response = workflow.run(
            f"""
            Research on this problem:
            Goal/problem: {problem_text}
            """
        )  
    st.markdown(response.content)
    

