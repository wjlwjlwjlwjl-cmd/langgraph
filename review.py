from langgraph.graph import StateGraph, START, END
from typing import Optional, Literal, TypedDict
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    desc: str
    option: Literal["通过", "拒绝"]
def call_node(state: State) -> Command[Literal["proceed", "reject"]]:
    message = "rm -rf /*"
    option = interrupt(message)
    return Command(goto="proceed" if option else "reject") # jump to specific position of nodes
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

checkpoint = InMemorySaver()
config = {
    "configurable": {
        "thread_id": "thread_id1"
    }
}
agent = agent_graph.compile(checkpointer=checkpoint)

ret = agent.invoke({}, config)["__interrupt__"][0].value
print(ret)
print(agent.invoke(Command(resume=True), config))