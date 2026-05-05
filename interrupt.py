from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    input: str
    output: str

def call_node(state: State):
    human = interrupt("accept?")
    if human == "yes":
        return {
            "output": "continue..."
        }
    else:
        return {
            "output": "stop"
        }

agent_graph = StateGraph(State)
agent_graph.add_node(call_node)
agent_graph.add_edge(START, "call_node")
agent_graph.add_edge("call_node", END) 

checkpoint = InMemorySaver()
agent = agent_graph.compile(checkpointer=checkpoint)

config = {"configurable": {"thread_id": "1"}}
print(agent.invoke({"input": "delete root directory"}, config)["__interrupt__"][0].value)
print(agent.invoke(Command(resume="yes"), config)["output"])
