from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class Node1Output(TypedDict):
    sensitive_data: str = Field(description="私密数据")
class Node2Input(TypedDict):
    sensitive_data: str = Field(description="私密数据输入")
class OverallState(TypedDict):
    data: str

def node1(state: OverallState) -> Node1Output:
    return {
        "sensitive_data": "敏感数据"
    }
def node2(state: Node2Input) -> OverallState:
    return {
         "data": "经过处理后的敏感数据"
    }
def final_output(state: OverallState):
    return {
        "data":  f"最后结果：{state["data"]}"
    }

agent_graph = StateGraph(OverallState)
agent_graph.add_sequence([node1, node2, final_output])

agent_graph.add_edge(START, "node1")
agent_graph.add_edge("final_output", END)

agent = agent_graph.compile()

print(agent.invoke({"data": "一个请求"}))