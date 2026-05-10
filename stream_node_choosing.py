from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
model = ChatOpenAI(model="qwen-turbo")
class State(TypedDict):
    content: str
    summary: str
    translate: str
def summary_node(state: State):
    content = state["content"]
    summary = model.invoke(f"给下面的内容做一个概括: {content}").content
    return {
        "summary": summary,
    }
def translate_node(state: State):
    content = state["content"]
    translate = model.invoke(f"将下面的内容翻译成英语: {content}").content
    return {
        "translate": translate
    }

agent = (StateGraph(State)
         .add_node(summary_node)
         .add_node(translate_node)
         .add_edge(START, "summary_node")
         .add_edge("summary_node", "translate_node")
         .add_edge("translate_node",  END)
         .compile())
for token_chunk, metadata in agent.stream({"content": "窗前明月光，疑是地上霜"}, stream_mode="messages"):
    node = metadata["langgraph_node"]
    if node == "translate_node":
        print(token_chunk.content, end="")
    else:
        print(node)
print()