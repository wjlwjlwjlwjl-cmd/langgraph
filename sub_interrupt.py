from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
import json

class SubState(TypedDict):
    msg: str

class ParentState(TypedDict):
    msg: str

def sub_node1(state: SubState):
    value = interrupt("continue?")
    return {"msg": value}


def sub_node2(state: SubState):
    value = interrupt("continue, sure?")
    return {"msg": value}

def parent_node(state: ParentState):
    return {"msg": "parent_node"}

sub_agent = (
    StateGraph(SubState)
    .add_node(sub_node1)
    .add_node(sub_node2)
    .add_edge(START, "sub_node1")
    .add_edge("sub_node1", "sub_node2")
    .compile()
)

checkpoint = InMemorySaver()
parent_agent = (
    StateGraph(ParentState)
    .add_node(parent_node)
    .add_node("sub_agent", sub_agent)
    .add_edge(START, "parent_node")
    .add_edge("parent_node", "sub_agent")
    .compile(checkpointer=checkpoint)
)

config = {"configurable": {"thread_id": "thread_id1"}}
parent_agent.invoke({}, config)
parent_state = parent_agent.get_state(config, subgraphs=True).tasks[0].state
print(parent_state)
print(parent_agent.invoke(Command(resume="yes"), config))