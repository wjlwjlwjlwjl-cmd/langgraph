from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    age: int | None


def get_and_check_age(state: State):
    age = state["age"]
    while True:
        if age is None:
            message = "请输入年龄（阿拉伯数字）："
            age = interrupt(message)
        if type(age) is int and age > 0:
            return {"age": age}
        else:
            message = "年龄非合理值"
            age = interrupt(message)


agent_graph = StateGraph(State)
agent_graph.add_node(get_and_check_age)
agent_graph.add_edge(START, "get_and_check_age")
agent_graph.add_edge("get_and_check_age", END)

checkpoint = InMemorySaver()
config = RunnableConfig({"configurable": {"thread_id": "thread_id1"}})
agent = agent_graph.compile(checkpointer=checkpoint)
first = agent.invoke({"age": None}, config)["__interrupt__"][0].value
print(first)
retry = agent.invoke(Command(resume="三十"), config)["__interrupt__"][0].value
print(retry)
final_result = agent.invoke(Command(resume=30), config)
print(final_result)
