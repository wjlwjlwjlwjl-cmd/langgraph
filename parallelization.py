from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    tech: str
    competitor: str
    user: str
    ret: str

def tech_node(state: State):
    return {
        "tech": "things related with tech..."
    }
def competitor_node(state: State):
    return {
        "competitor": "things related with competitors..."
    }
def user_node(state: State):
    return {
        "user": "things related with user"
    }
def output_node(state: State):
    return {
        "ret": state["tech"] + state["competitor"] + state["user"]
    }
agent_graph = StateGraph(State)
agent_graph.add_node(tech_node)
agent_graph.add_node(competitor_node)
agent_graph.add_node(user_node)
agent_graph.add_node(output_node)

agent_graph.add_edge(START, "tech_node")
agent_graph.add_edge(START, "competitor_node")
agent_graph.add_edge(START, "user_node")

agent_graph.add_edge("tech_node", "output_node")
agent_graph.add_edge("competitor_node", "output_node")
agent_graph.add_edge("user_node", "output_node")
agent_graph.add_edge("output_node", END)

agent = agent_graph.compile()

print(agent.invoke({})["ret"])