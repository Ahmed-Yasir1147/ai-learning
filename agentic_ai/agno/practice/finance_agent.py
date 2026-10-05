from agno.agent import Agent
from dotenv import load_dotenv
from google import genai
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

def build_agent(model_id: str = "gemini-3.5-flash-lite", ) -> Agent:
    agent = Agent(
        model=Gemini(id=model_id),
        tools=[
            YFinanceTools(),
            DuckDuckGoTools()
        ],
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals and response with correct information in concise manner.",
        instructions=["Format your response using markdown and use tables to display data where possible."],      
        debug_mode=True
    )
    return agent

agent = build_agent()
agent.print_response("Give stock price for Microsoft and give analyst recommendation")