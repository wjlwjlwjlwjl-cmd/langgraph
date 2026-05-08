from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
import operator

model = ChatOpenAI(model="qwen-turbo")
class MessageState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    user_name: str
@dataclass
class ContextSchema:
    user_id: str = ""
@tool
def search(runtime: ToolRuntime[ContextSchema]):
    """调用搜索工具，用来查询天气信息"""
    state = runtime.state
    context = runtime.context
    user_id = context.user_id
    user_name = state["user_name"]
    print(f"{user_name}: {user_id}, call the tool")
    return f"{user_name}您好，今日上海天气阴天，较昨日降温："

def llm_node(state: MessageState):
    messages = state["messages"]
    model_with_tool = model.bind_tools([search])
    return {
        "messages": [model_with_tool.invoke([
            SystemMessage(content="你是一个天气助手，需要时使用我提供的搜索工具进行搜索"),
        ] + messages)]
    }
agent_graph = StateGraph(MessageState, context_schema=ContextSchema)
agent_graph.add_node(llm_node)
agent_graph.add_node("tool_node", ToolNode([search]))
agent_graph.add_edge(START, "llm_node")
agent_graph.add_conditional_edges(
    "llm_node",
    tools_condition,
    {
        "tools": "tool_node", 
        "__end__": END
    }
)
agent = agent_graph.compile()
print(agent.invoke(MessageState(messages=[HumanMessage(content="今日上海天气如何？")], user_name="wjl"), context=ContextSchema(user_id="user_id1")))
for chunk in agent.stream(MessageState(messages=[HumanMessage(content="上海今日天气如何？")], user_name="wjl"), context=ContextSchema(user_id="user_id1")):
    for name, update in chunk.items():
        update["messages"][-1].pretty_print()
