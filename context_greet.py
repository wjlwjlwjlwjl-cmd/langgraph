from langgraph.graph import StateGraph,START, END
from dataclasses import dataclass
from typing import TypedDict
from langgraph.runtime import Runtime

# 定义运行时上下文
@dataclass
class ContextSchema:
    language: str = "en"

class State(TypedDict):
    greet: str    
    user_name: str

# 在节点中访问上下文
def greet_node(state: State, runtime: Runtime[ContextSchema]):
    user_name = state.get("user_name", "")
    language = runtime.context.language
    if language == "en":
        greet = f"hello {user_name}"
    else:
        greet = f"你好 {user_name}"
    return {
        "greet": greet
    }

# 在图中使用上下文
agent_graph = StateGraph(State, context_schema=ContextSchema)
agent_graph.add_node(greet_node)
agent_graph.add_edge(START, "greet_node")
agent_graph.add_edge("greet_node", END)

agent = agent_graph.compile()

# 图运行时传入时传入上下文
state = State({"user_name": "wjl", "greet": ""})
context = ContextSchema(language="en")
print(agent.invoke(state, context=context)["greet"])
