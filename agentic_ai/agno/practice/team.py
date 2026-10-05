from typing import List

from agno.agent import Agent
from dotenv import load_dotenv
from google import genai
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team

load_dotenv()

def build_team(name: str, members: List[Agent], instructions: str = "", gemini_model_id: str = "gemini-3.5-flash-lite"):
    team = Team(
        name=name,
        members=members,
        model=Gemini(
            id=gemini_model_id
        ),
        markdown=True,
        show_members_responses=True,
        instructions=instructions
    )
    return team

team = build_team(
    "Translation team",
    [
        Agent(name="English Agent", role="Translate into English"),
        Agent(name="Urdu Agent", role="Translate into Urdu"),
        Agent(name="Punjabi Agent", role="Translate into Punjabi while using Urdu script")
    ],
    "Translate using all agents if translation language is not mentioned otherwise only in given language"
)

team.print_response("Trnaslate into Urdu: I will kick your ass!")
