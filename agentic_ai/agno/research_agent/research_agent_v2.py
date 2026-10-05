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
from agno.workflow.loop import Loop
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

HISTORY_RUNS = 1

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
        "You receive the research plan and execute it. You follow various steps and tools suggested to get information.",
        "You present raw information in below format: Original query and collection of each finding, relevant information and references & evidence for it."
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
        """,
        "The report outlines is not strict rule it can vary depending on problem and info gathered!"
    ]
)

class EvaluationResponse(BaseModel):
    ok: bool = Field(description="If given information is enough for goal/problem")
    reasoning: str = Field(description="Tell why you accepted or rejected it briefly")

information_evaluator = Agent(
    name="Information Evaluator",
    model=Gemini(
        id="gemini-3.5-flash-lite"
    ),
    instructions=[
        "You are given information gathered for goal/problem embedded in the content itself. You have to analyze these.",
        "You have to tell if given retrieved information broadly answered the original question or we need more information",
        "Answer strictly in => true: if its enough, false: if more info is needed",
        "Evaluate strictly when it comes to broadness & quality of information gathered but don't reject unnecessarily"
    ],
    output_schema=EvaluationResponse
)

def planning_evaluator(outputs: list[StepOutput]) -> bool:
    prompt = f"""Tell if given info is enough to create structured output: 
    Information: {outputs[0].content}
    """

    evaluation = information_evaluator.run(
        prompt,
    )

    if not evaluation.content.ok:
        print(f"*************\n Bad research: {evaluation.content.reasoning} \n*****************")
    else: 
        print(f"**************\n Good research: {evaluation.content.reasoning} \n*****************")

    return evaluation.content.ok

def build_research_workflow() -> Workflow:
    return Workflow(
        name="Researcher",
        description="You research and provide structured report about given topic/goal",
        steps=[
            # Combining plan & retrieval because they are interconnected & their success
            # or failure is coupled
            Loop(
                name="Planning & Retrieval Loop",
                max_iterations=3,
                steps=[
                    Step(
                        name="Planning",
                        agent=research_planner,
                    ),
                    Step(
                        name="Searching & Retrieval",
                        agent=information_retriever,
                    ),
                ],
                end_condition=planning_evaluator
            ),
            Step(
                name="Report Generation",
                agent=report_generator
            )
        ],
        add_workflow_history_to_steps=True,
        num_history_runs=HISTORY_RUNS,
    )

