from typing_extensions import Annotated, TypedDict
from langgraph.types import Overwrite
from langgraph.graph import StateGraph, START, END
from typing import List
import operator


class State(TypedDict):
    messages: Annotated[List[str], operator.add]


agent_graph = StateGraph(State)


def add_message(state: State):
    return {"messages": ["first message"]}


def replace_message(state: State):
    return {"messages": Overwrite(["replacement message"])}


agent_graph.add_node(add_message)
agent_graph.add_node(replace_message)

agent_graph.add_edge(START, "add_message")
agent_graph.add_edge("add_message", "replace_message")
agent_graph.add_edge("replace_message", END)

agent = agent_graph.compile()
ret = agent.invoke({"messages": ["a message"]})
print(ret)

