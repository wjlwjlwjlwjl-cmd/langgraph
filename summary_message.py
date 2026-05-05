from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOpenAI(
    model="qwen-turbo"
)
class State(MessagesState):
    summary: str

def call_node(state: State):
    response = model.invoke([state.get("summary", "")] + state["messages"])
    return {
        "messages": [response]
    }
def summary_node(state: State):
    summary = state.get("summary", "")

    if summary:
        prompt = state["messages"] + [HumanMessage(content=f"上面是所有的聊天信息，请根据这些信息扩展如下摘要,并且不要在加上新信息的同时丢失老信息：{summary}")]
    else:
        prompt = state["messages"] + [HumanMessage(content="请给上面的聊天信息生成一份摘要")]

    delete_message = [RemoveMessage(id=message.id) for message in state["messages"][:-2]]
    new_summary = model.invoke(prompt).content
    return {
        "messages": delete_message,
        "summary": new_summary
    }
agent_graph = StateGraph(State)
agent_graph.add_node(call_node)
agent_graph.add_node(summary_node)
agent_graph.add_edge(START, "call_node")
agent_graph.add_edge("call_node", "summary_node")
agent_graph.add_edge("summary_node", END)

checkpoint = InMemorySaver()
agent = agent_graph.compile(checkpointer=checkpoint)

config = {"configurable": {"thread_id": "2"}}
agent.invoke({"messages": "我叫小明"}, config)
agent.invoke({"messages": "我喜欢小猫，请帮我写一首小猫的诗"}, config)
agent.invoke({"messages": "我也喜欢小狗，现在请帮我写一首小狗的诗"}, config)
print(agent.invoke({"messages": "我叫什么？"}, config)["messages"][-1].content)
print(agent.invoke({"messages": "我叫什么？"}, config)["summary"])
