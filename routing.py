from langgraph.graph import StateGraph
from langgraph.constants import START, END
from pydantic import Field
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = "qwen-turbo"
)

class Route(TypedDict):
    step: Literal["before_sale", "after_sale", "technical"] = Field("根据请求类型选择相应的处理方式：售前、售后、技术问题")

class State(TypedDict):
    input: str
    step : str
    output: str

def model_route(state: State):
    step = model.with_structured_output(Route).invoke(state["input"])["step"]
    return {
        "step": step
    }
def before_sale_node(state: State):
    return {
        "output": "before_sale_answer"
    }
def after_sale_node(state: State):
    return {
        "output": "after_sale_answer"
    }
def technical_node(state: State):
    return {
        "output": "technical_answer"
    }

def route_function(state: State):
    if state["step"] == "before_sale":
        return "before_sale_node"
    elif state["step"] == "after_sale":
        return "after_sale_node"
    else:
        return "technical_node"

agent_graph = StateGraph(State)
agent_graph.add_node(model_route)
agent_graph.add_node(before_sale_node)
agent_graph.add_node(after_sale_node)
agent_graph.add_node(technical_node)

agent_graph.add_edge(START, "model_route")
agent_graph.add_conditional_edges(
    "model_route",
    route_function
)
agent_graph.add_edge("before_sale_node", END)
agent_graph.add_edge("after_sale_node", END)
agent_graph.add_edge("technical_node", END)

agent = agent_graph.compile()

questions = [
    "我购买的产品有质量问题，需要退货", # 售后问题
    "这个软件安装后⽆法正常运⾏，报错代码0x80070005", # 技术问题
    "请问你们的售后服务政策是什么", # 售前咨询
    "我的订单已经发货但还没收到", # 售后问题
    "如何配置数据库连接参数"
]
for question in questions:
    print(agent.invoke({"input": question})["output"])
    print("*" * 30)
