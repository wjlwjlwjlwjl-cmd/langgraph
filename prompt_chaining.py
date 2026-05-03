# 提示链模式
import plistlib
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import Field

model = ChatOpenAI(
    model = "qwen-turbo"
)

class InputState(TypedDict):
    topic: Annotated[str,  Field(description="文章主题")]
class OutputState(TypedDict):
    result: Annotated[str, Field(description="文章最后正文")]
class OverallState(InputState, OutputState):
    outline: Annotated[str, Field(description="文章大纲")]
    draft: Annotated[str, Field(description="文章初稿")]
    polished_draft: Annotated[str, Field(description="润色过的文章")]

# 大纲生成节点
PROMPT_1 = (
    "根据主题⽣成⽂章⼤纲。\n"
    "主题：{topic}\n"
    "要求："
    "1.只需两个最核⼼标题"
    "2.不⽤进⾏说明，只返回最终⼤纲"
)
def outline_node(state: InputState) -> OverallState:
    print("生成大纲中...")
    outline = model.invoke(PROMPT_1.format(topic=state["topic"])).content
    return {
        "topic": InputState["topic"],
        "outline": outline
    }

PROMPT_2 = (
"根据以下内容⽣成⽂章完整初稿。\n"
"主题：{topic}\n"
"⼤纲: "
"{outline}\n"
"要求："
"1.每个标题下，最多使⽤三句话的内容即可"
"2.不⽤进⾏说明，只返回最终结果"
)
def draft_node(state: OverallState):
    print("生成初稿中...")
    draft = model.invoke(PROMPT_2.format(topic=state["topic"], outline=state["outline"])).content
    return {
        "topic": state["topic"],
        "draft": draft,
        "outline": state["outline"]
    }

PROMPT_3 = (
    "根据⽂章初稿进⾏润⾊。\n"
    "主题：{topic}\n"
    "初稿: "
    "{draft}\n"
    "要求："
    "1.润⾊后，⽂章不能太⻓"
)
def polish_node(state: OverallState):
    print("润色初稿中...")
    polished_draft = model.invoke(PROMPT_3.format(topic=state["topic"], draft=state["draft"])).content
    return {
        "topic": state["topic"],
        "outline": state["outline"],
        "draft": state["draft"],
        "polished_draft": polished_draft
    }

PROMPT_4 = (
    "根据润⾊版⽂章，⽣成⽂章终稿。\n"
    "主题：{topic}\n"
    "⼤纲: "
    "{outline}\n"
    "润⾊版⽂章: "
    "{polished_draft}\n"
)
def output_node(state: OverallState) -> OutputState:
    print("整理最终输出中...")
    result = model.invoke(PROMPT_4.format(topic=state["topic"], outline=state["outline"], polished_draft=state["polished_draft"])).content
    return {
        "result": result
    }

agent_graph = StateGraph(OverallState)
agent_graph.add_sequence([outline_node, draft_node, polish_node, output_node])
agent_graph.add_edge(START, "outline_node")
agent_graph.add_edge("output_node", END)

agent = agent_graph.compile()

print(agent.invoke({"topic": "卖核弹的小女孩"})["result"])