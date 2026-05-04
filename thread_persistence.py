from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import TypedDict, List, Annotated
import operator
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite

DB_URL = "postgres://postgres:bit@localhost:5432/postgres"
with PostgresSaver.from_conn_string(DB_URL) as check_point:
    check_point.setup()

    # 定义聊天模型和工具
    model = ChatOpenAI(model="qwen-turbo")

    @tool
    def add(a: int, b: int):
        """
        所有加法计算都通过add工具进行
        Args:
            a: int 第一个参数
            b: int 第二个参数
        Return:
            int 加法的结果
        """
        return a + b

    tools = [add]
    model_with_tool = model.bind_tools(tools)

    # 定义状态
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], operator.add]
        llm_cnt: int

    # 构建图
    state_graph = StateGraph(State)

    # 定义聊天模型节点
    def llm_node(state: State):
        result = [
            model_with_tool.invoke(
                [
                    SystemMessage("你是一个数学大师，支持调用工具计算")
                ] + state["messages"]
            )
        ]
        return {
            "messages": result,
            "llm_cnt": state.get("llm_cnt", 0) + 1
        }


    # 定义工具调用节点
    tool_by_name = {tool.name: tool for tool in tools}

    def tool_node(state: State):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tool_by_name[tool_call["name"]]
            obs = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))
        return {
            "messages": result
        }

    # 添加节点
    state_graph.add_node("llm_node", llm_node)
    state_graph.add_node("tool_node", tool_node)
    # 添加边
    state_graph.add_edge(START, "llm_node")

    def should_continue(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    state_graph.add_edge("tool_node", "llm_node")
    state_graph.add_conditional_edges(
        "llm_node",
        should_continue
    )
    final_graph = state_graph.compile(checkpointer=check_point)

    config={"configurable": {"thread_id": 1}}
    result = final_graph.invoke(
        {
            "messages": [
                HumanMessage(content="1 + 1 等于多少？")
            ]
        },
        config
    )
    print(f"共调用了{result["llm_cnt"]}次大模型")
    for msg in result["messages"]:
        msg.pretty_print()

    # 获取所有历史记录
    selected_state = None
    for state in final_graph.get_state_history(config):
        if len(state.values["messages"]) == 1:
            selected_state = state
    new_state = final_graph.update_state(selected_state.config, {"messages": Overwrite([HumanMessage(content="2 + 2 等于几？")])})
    print(final_graph.invoke(None, config=new_state)["messages"][-1].content)