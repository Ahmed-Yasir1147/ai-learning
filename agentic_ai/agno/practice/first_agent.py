from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.github import GithubTools

load_dotenv()

def travel_tool(source: str, destination: str):

    """
    Use this when user is asking about price or time to go from one place to another.
    args:
        source: where he starts from
        destination: where he wants to go
    """

    return {
        "source": source,
        "destination": destination,
        "cost": "500$",
        "time": "9 hours"
    }

def build_agent(model_id: str = "gemini-3.5-flash-lite", ) -> Agent:
    agent = Agent(
        model=Gemini(id=model_id),
        description="You are a helfpul expert Travel Agent.",
        markdown=True,
        add_datetime_to_context=True,
        tools=[
            GithubTools(
                include_tools=[
                    "get_repository",
                    "list_repositories",
                    "get_pull_requests",
                    "list_issues",
                    "list_branches"
                ]
            )
        ]
    )
    return agent

agent = build_agent()

agent.print_response("Tell me about repo: Sticky-Notes-Android What is this about? I am trying to test if Github tool is working so tell if you were able to access this from GITHUB_ACCESS_TOKEN")