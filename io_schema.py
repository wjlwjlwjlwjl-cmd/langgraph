from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field

class InputSchema(TypedDict):
    question: Annotated[str, Field(description="问题描述")]

class OutputSchema(TypedDict):
    answer: Annotated[str, Field(description="答案")]

class ProcessState(InputSchema, OutputSchema):
    pass

agent_graph = StateGraph(
    ProcessState,
    input_schema=InputSchema,
    output_schema=OutputSchema
)

def answer_node(state: ProcessState):
    return {
        "answer": f"answer of {state['question']}",
    }

agent_graph.add_node(answer_node)
agent_graph.add_edge(START, "answer_node")
agent_graph.add_edge("answer_node", END)

agent = agent_graph.compile()
print(agent.invoke({"question": "a test question"}))