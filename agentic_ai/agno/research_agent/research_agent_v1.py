# Basic Research Planner 
# Three steps Workflow:
# 1. Plan
# 2. Retrieve
# 3. Organize

from agno.agent import Agent
from agno.models.google import Gemini
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

HISTORY_RUNS = 3

research_planner = Agent(
    name="Research-Planner",
    model=Gemini(
        id="gemini-3.5-flash-lite"
    ),
    instructions=[
        "You will receive a vague or good prompt, you have to understand the goal, scope and intent of request",
        "You have to provide a plan which will guide another LLM agent to fetch information and use relevant tools"
    ]
)

information_retriever = Agent(
    name="Information-Retriever",
    model=Gemini(
        id="gemini-3.5-flash-lite"
    ),
    instructions=[
        "You receive the research plan and execute it. You follow various steps and tools suggested to get information."
        "You present this information in organized form with clear headings mentioned which part contains what."
    ]
)

report_generator = Agent(
    name="Report-Generator",
    model=Gemini(
        id="gemini-3.5-flash-lite"
    ),
    instructions=[
        "You receive extracted information about research goal and understand it.",
        """You create a structured report containing:  
            1. Research question
            2. Executive summary
            3. Findings
            4. Evidence for each major finding
            5. Conflicting evidence
            6. Analysis
            7. Limitations
            8. References
        """
    ],
    markdown=True
)

research_workflow = Workflow(
    name="Researcher",
    description="You research and provide structured report about given topic/goal",
    steps=[
        Step(
            name="Planning",
            agent=research_planner,
        ),
        Step(
            name="Searching & Retrieval",
            agent=information_retriever,
        ),
        Step(
            name="Report Generation",
            agent=report_generator
        )
    ],
    add_workflow_history_to_steps=True,
    num_history_runs=HISTORY_RUNS,
    debug_mode=True
)

research_workflow.run("Is doing Computer Science still worth it? Answer with research findings.")