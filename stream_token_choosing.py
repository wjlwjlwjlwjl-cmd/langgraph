from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict

model_joke = ChatOpenAI(
    model="qwen-turbo",
    model_kwargs={"tags": ["joke"]}
)
model_poem = ChatOpenAI(
    model="qwen-turbo",
    model_kwargs={"tags": ["poem"]}
)
class State(TypedDict):
    topic: str
    joke: str
    poem: str
def call_node(state: State):
    topic = state["topic"]
    joke = model_joke.invoke(f"write to a joke about {topic}").content
    poem = model_joke.invoke(f"write to a poem about {topic}").content
    return {
        "joke": joke,
        "porm": poem
    }
agent = (StateGraph(State)
               .add_node(call_node)
               .add_edge(START, "call_node")
               .add_edge("call_node",  END)
               .compile())
for token_chunk, metadata in agent.stream({"topic": "programmer"}, stream_mode="messages"):
    tag = metadata["tags"][0]
    if tag == "joke":
        print(token_chunk["joke"])
    