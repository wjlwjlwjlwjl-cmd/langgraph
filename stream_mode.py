def values_and_updates():
    from langgraph.graph import StateGraph, END, START
    import time

    class State(dict):
        topic: str
        joke: str

    def generate_node(state):
        time.sleep(2)
        return {"joke": "a joke related with topic: " + state["topic"]}
    def refine_node(state):
        time.sleep(2)
        return {"joke": state["joke"] + " --- refined"}

    agent = (
        StateGraph(State)
        .add_node(generate_node)
        .add_node(refine_node)
        .add_edge(START, "generate_node")
        .add_edge("generate_node", "refine_node")
        .add_edge("refine_node",  END)
        .compile()
    )

    # values 展示所有流式输出内容
    for chunk in agent.stream(
        {
            "topic": "cat"
        },
        stream_mode="values"
    ):
        print(chunk)

    # updates 只流式输出所有更新内容
    for chunk in agent.stream(
        {
            "topic": "cat"
        },
        stream_mode="updates"
    ):
        print(chunk)

def custom():
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_openai import ChatOpenAI
    from typing import TypedDict, Annotated, List
    from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
    from langchain.tools import tool, ToolRuntime
    from dataclasses import dataclass
    from langgraph.config import get_stream_writer
    import operator
    import time

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
        writer = get_stream_writer()

        writer({
            "tool_name": "search",
            "info": "begin to use tool"
        })

        time.sleep(2)
        writer({
            "result": "success",
            "tool_name": "search"
        })

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
    for chunk in agent.stream(MessageState(messages=[HumanMessage(content="上海今日天气如何？")], user_name="wjl"), context=ContextSchema(user_id="user_id1"), stream_mode="custom"):
        print(chunk)

def messages():      
    from langgraph.graph import StateGraph, START, END
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from typing import TypedDict
    model = ChatOpenAI(model = "qwen-turbo")
    class State(TypedDict):
        input: str
        output: str
    def answer_node(state: State):
        prompt = ChatPromptTemplate.from_messages(
            [("system", "you are a super good writer"),
            ("user", "write me a novel about {state} no less than 1000 words")
        ])
        return {
            "output": model.invoke(prompt.invoke({"state": state["input"]}))
        }
    agent = (
        StateGraph(State)
        .add_node(answer_node)
        .add_edge(START, "answer_node")
        .add_edge("answer_node", END)
        .compile()
    )
    for chunk, metadata in agent.stream(
        input={
            "input": "love",
            "output": ""
        },
        stream_mode="messages"
    ):
        print(chunk.content, end = "", flush=True)
messages()