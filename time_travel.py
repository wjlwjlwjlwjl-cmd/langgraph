from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, NotRequired
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

model = ChatOpenAI(
    model = "qwen-turbo"
)

class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]

def topic_node(state: State):
    request = "give me a topic of joke"
    topic = model.invoke(request).content
    return {
        "topic": topic
    }
def joke_node(state: State):
    topic = state.get("topic", "")
    request = f"give me a joke base on the topic: {topic}"
    joke = model.invoke(request).content
    return {
        "joke": joke
    }

agent_graph = StateGraph(State)
agent_graph.add_node(joke_node)
agent_graph.add_node(topic_node)
agent_graph.add_edge(START, "topic_node")
agent_graph.add_edge("topic_node", "joke_node")
agent_graph.add_edge("joke_node", END)

checkpoint = InMemorySaver()
agent = agent_graph.compile(checkpointer=checkpoint)

config = RunnableConfig({"configurable": {"thread_id": "thread_id1"}})
agent.invoke({}, config)

history = list(agent.get_state_history(config))
target = history[1]
new_state = agent.update_state(target.config, {"topic": "the instresting things about programmar"})
print(agent.invoke(None, new_state)["joke"])
