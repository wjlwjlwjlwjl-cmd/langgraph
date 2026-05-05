from langgraph.graph import StateGraph, START, END
from typing import Optional, Literal, TypedDict
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    decision: str
    option: Optional[Literal["通过", "拒绝"]]

def call_node(state: State) -> Command[Literal["proceed", "reject"]]:
    decision = "给张三转十万元"
    human = interrupt(decision)
    return Command(goto="proceed" if human else "reject")
def proceed(state: State):
    return {
        "option": "通过"
    }
def reject(state: State):
    return {
        "option": "拒绝"
    }

agent_graph = StateGraph(State)
agent_graph.add_node(call_node)
agent_graph.add_node(proceed)
agent_graph.add_node(reject)
agent_graph.add_edge(START, "call_node")
agent_graph.add_edge("proceed", END)
agent_graph.add_edge("reject", END)

config = {"configurable": {"thread_id": "thread_id1"}}
checkpoint = InMemorySaver()
agent = agent_graph.compile(checkpointer=checkpoint)
ret = agent.invoke({}, config)
print(ret["__interrupt__"][0].value)
print(agent.invoke(Command(resume=True), config))