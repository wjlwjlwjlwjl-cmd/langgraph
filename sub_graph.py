from langgraph.graph import StateGraph, START, END
from typing import TypedDict


def mode1():
    """由节点调用主图，父图、子图之间不共享状态"""

    class SubState(TypedDict):
        msg: str

    class ParentState(TypedDict):
        msg: str

    def sub_node1(state: SubState):
        return {"msg": "sub_node1"}

    def sub_node2(state: SubState):
        return {"msg": "sub_node2"}

    sub_agent = (
        StateGraph(SubState)
        .add_node(sub_node1)
        .add_node(sub_node2)
        .add_edge(START, "sub_node1")
        .add_edge("sub_node1", "sub_node2")
        .add_edge("sub_node2", END)
        .compile()
    )

    def parent_node1(state: SubState):
        msg = sub_agent.invoke({})["msg"]
        return {"msg": "parent_node1" + msg}

    def parent_node2(state: SubState):
        return {"msg": "parent_node2"}

    parent_agent = (
        StateGraph(ParentState)
        .add_node(parent_node1)
        .add_node(parent_node2)
        .add_edge(START, "parent_node1")
        .add_edge("parent_node1", "parent_node2")
        .add_edge("parent_node2", END)
        .compile()
    )
    for token_chunk in parent_agent.stream({}, subgraphs=True):
        print(token_chunk)


def mode2():
    """将子图作为节点，子图、父图之间共享状态"""

    class SubState(TypedDict):
        parent: str
        sub: str

    class ParentState(TypedDict):
        parent: str

    def sub_node1(state: SubState):
        return {"parent": "sub as parent's node", "sub": "sub_node1"}

    def sub_node2(state: SubState):
        return {"parent": "sub as parent's node", "sub": "sub_node2"}

    sub_agent = (
        StateGraph(SubState)
        .add_node(sub_node1)
        .add_node(sub_node2)
        .add_edge("sub_node1", "sub_node2")
        .add_edge(START, "sub_node1")
        .compile()
    )

    def parent_node(state: ParentState):
        return {"parent": "the parent's node"}

    parent_agent = (
        StateGraph(ParentState)
        .add_node("node1", parent_node)
        .add_node("node2", sub_agent)
        .add_edge(START, "node1")
        .add_edge("node1", "node2")
        .compile()
    )

    for token_chunk in parent_agent.stream({}, subgraphs=True):
        print(token_chunk)


mode2()
