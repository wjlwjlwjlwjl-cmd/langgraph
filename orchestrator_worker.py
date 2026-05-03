from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from typing import List
from langgraph.types import Send
import operator

model = ChatOpenAI(
    model="qwen-turbo"
)
class Section(TypedDict):
    name: str
    description: str
class Sections(TypedDict):
    sections: List[Section]
class State(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[List[str], operator.add]
    result: str

# 协调者
def orchestrator(state: State):
    planner = model.with_structured_output(Sections)
    sections = planner.invoke(f"请你做一篇关于{state["topic"]}的报告大纲，一共三个章节")
    return {
        "sections": sections["sections"]
    }
# 任务分配者
def mission_assigner(state: State):
    sections = state["sections"]
    workers = []
    for section in sections:
        workers.append(Send("llm_node", {"section": section}))
    return workers
# 大模型执行任务
def llm_node(state: State):
    section = state["section"]
    name = section["name"]
    description = section["description"]
    result = model.invoke(f"编写报告章节：{name}, 编写内容要求：{description}").content
    return {
        "completed_sections": [result]
    }
# 合成总结果
def synthesizer(state: State):
    content = "\n\n" .join(state["completed_sections"])
    result = f"# {state['topic']}\n\n{content}"
    return {
        "result": result
    }
agent_graph = StateGraph(State)
agent_graph.add_node(orchestrator)
agent_graph.add_node(mission_assigner)
agent_graph.add_node(llm_node)
agent_graph.add_node(synthesizer)

agent_graph.add_edge(START, "orchestrator")
agent_graph.add_conditional_edges(
    "orchestrator",
    mission_assigner
)
agent_graph.add_edge("llm_node", "synthesizer")
agent_graph.add_edge("synthesizer", END)

agent = agent_graph.compile()

print((agent.invoke({"topic": "中国近代史"}))["result"])