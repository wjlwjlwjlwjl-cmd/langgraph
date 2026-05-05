from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

model = ChatOpenAI(
    model="qwen-turbo"
)
def call_node(state: MessagesState):
    """messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=10480,
        start_on="human",
        end_on=("human", "tool")
    )
    response = model.invoke(messages)
    """

    # response = model.invoke(state["messages"])

    # 删除消息
    # RemoveMessage 是 LangGraph 中的一条指令
    messages = state["messages"]
    print(len(messages))
    delete_message = []
    if len(messages) >= 5:
        #delete_message = [RemoveMessage(id=m.id) for m in messages[:6]]
        delete_message = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    if len(messages) == 7:
        print([m.content for m in messages])
    response = model.invoke(state["messages"])
    return {
        "messages": delete_message + [response],
    }
in_memory_saver = InMemorySaver()
agent_graph = StateGraph(MessagesState)
agent_graph.add_node(call_node)
agent_graph.add_edge(START, "call_node")
agent_graph.add_edge("call_node", END)

config = {"configurable": {"thread_id": "2"}}
agent = agent_graph.compile(checkpointer=in_memory_saver)
agent.invoke({"messages": "我叫小明"}, config)
agent.invoke({"messages": "我喜欢小猫，请帮我写一首小猫的诗"}, config)
agent.invoke({"messages": "我也喜欢小狗，现在请帮我写一首小狗的诗"}, config)
print(agent.invoke({"messages": "我叫什么？"}, config)["messages"][-1].content)