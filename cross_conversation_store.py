from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from pydantic import Field, BaseModel
from langgraph.store.memory import InMemoryStore, BaseStore
import uuid

model = ChatOpenAI(
    model="qwen-turbo"
)
class Person(BaseModel):
    name: Annotated[str, Field(default=None, description="人物的姓名")]
    height_in_meter: Annotated[str, Field(default=None, description="人物以米为单位的身高")]
    like_food: Annotated[str, Field(default=None, description="喜欢吃的食物")]

model_with_structured_output = model.with_structured_output(Person)

class State(TypedDict):
    messages: Annotated[str, Field(description="任务输入的描述")]
store = InMemoryStore()

def get_person_by_llm(state: State, config: RunnableConfig, store: BaseStore):
    info = model_with_structured_output.invoke(
        [
            SystemMessage(
                content="请严格从用户输入中提取以下信息，只提取对应内容，不要乱填：\n"
                        "name：姓名\n"
                        "height_in_meter：身高\n"
                        "like_food：喜欢的食物\n"
                        "没有则返回null，必须返回合法JSON"),
            HumanMessage(state["messages"])
        ]
    )
    name = info.name
    height_in_meter = info.height_in_meter
    like_food = info.like_food
    if height_in_meter == "null":
        height_in_meter = None
    if like_food == "null":
        like_food = None

    namespace1 = (name, "info")
    namespace2 = (name, "preference")

    if not store.search(namespace1) and height_in_meter is not None:
        store.put(namespace1, str(uuid.uuid4()), height_in_meter)
    if not store.search(namespace2) and like_food is not None:
        store.put(namespace=namespace2, key=str(uuid.uuid4()), value=like_food)

    if not height_in_meter:
        height_in_meter = store.search(namespace1)[0].value
    if not like_food:
        like_food = store.search(namespace2)[0].value

    return {
        "messages": name + ": " + height_in_meter + ": " + like_food
    }
agent_graph = StateGraph(State)
agent_graph.add_node(get_person_by_llm)

agent_graph.add_edge(START, "get_person_by_llm")
agent_graph.add_edge("get_person_by_llm", END)

agent = agent_graph.compile(store=store)

config = {
    "configurable": {"thread_id": "thread-1"}
}
print(agent.invoke({"messages": "我叫李华，身高一米八，爱吃汉堡"}, config=config))

config = {
    "configurable": {"thread_id": "thread-2"}
}
print(agent.invoke({"messages": "我叫李华，我喜欢吃什么?"}, config=config))
