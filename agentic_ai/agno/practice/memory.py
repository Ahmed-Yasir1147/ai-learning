from agno.agent import Agent
from dotenv import load_dotenv
from google import genai
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from rich.pretty import pprint

load_dotenv()

db = SqliteDb(db_file="agno.db")

def build_agent(db: SqliteDb, gemini_model_id: str = "gemini-3.5-flash-lite", description: str = "", instructions:  str = "") -> Agent:
    agent = Agent(
        model=Gemini(id=gemini_model_id),
        description=description,
        instructions=instructions,   
        markdown=True,
        add_history_to_context=True,
        update_memory_on_run=True,
        db=db
    )
    return agent

agent = build_agent(
    db=db,
    description="You are a helpful bot"
)
user_id_1 = "Ahmed"
user_id_2 = "Talha"
agent.print_response("Can you recall what topis we have discussed previously", user_id=user_id_1)


print("Memories:")
pprint(agent.get_user_memories(user_id=user_id_1))