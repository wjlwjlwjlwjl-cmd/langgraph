from langchain_community.callbacks import HumanApprovalCallbackHandler
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_community.document_loaders import TextLoader
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import Field, BaseModel

model = ChatOpenAI(
    model = "qwen-turbo"
)
embedding = OllamaEmbeddings(
    model = "all-minilm"
)
documents = ["bite_cpp.md", "bite_java.md", "bite_test.md", "bite_info.md"]
splits = []
spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
for document in documents:
    loader = TextLoader(document, encoding="utf-8")
    data = loader.load()
    splits.append(spliter.split_documents(data))
vector_store = InMemoryVectorStore(
    embedding=embedding
)
for doc in splits:
    vector_store.add_documents(doc)
retriever = vector_store.as_retriever()
tool_retriever = create_retriever_tool(retriever, name="文档检索工具", description="当查询和比特有关的内容时，使用这个工具查询文档")
model_with_tool = model.bind_tools([tool_retriever], tool_choice="文档检索工具")

agent_graph = StateGraph(MessagesState)

# 调用工具或生成
def generate_resp_or_invoke_tool(state: MessagesState):
    response = model_with_tool.invoke(state["messages"])
    return {
        "messages": [response]
    }

# 工具调用节点
tool_node = ToolNode([tool_retriever])

# 重写问题、重新检索
REWRITE_PROMPT = (
    "查看输⼊并尝试推断潜在的语义意图/含义。\n"
    "这是最初的问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "提出⼀个改进后的问题："
)
def rewrite_node(state: MessagesState):
    message = state["messages"]
    question = message[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke(prompt)
    return {
        "messages": [response]
    }

# 最后生成结果
GENERATE_PROMPT = (
    "你是负责回答问题的助⼿。 "
    "使⽤以下检索到的上下⽂⽚段来回答问题。 "
    "如果你不知道答案，就说你不知道。 "
    "最多只⽤三句话，回答要简明扼要。\n"
    "Question: {question} \n"
    "Context: {context}"
)
def answer_node(state: MessagesState):
    message = state["messages"]
    question = message[0].content
    context = message[-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke(prompt)
    return {
        "messages": [response]
    }

agent_graph.add_node(generate_resp_or_invoke_tool)
agent_graph.add_node("tool_node", tool_node)
agent_graph.add_node(rewrite_node)
agent_graph.add_node(answer_node)

agent_graph.add_edge(START, "generate_resp_or_invoke_tool")
agent_graph.add_conditional_edges(
    "generate_resp_or_invoke_tool",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)

class similar_ret(BaseModel):
    ret: str = Field(description="用来存放二元分数，相关为'yes'，否则为'no'")
GRADE_PROMPT = (
    "你是⼀个评分员，评估检索到的⽂档与⽤⼾问题的相关性。 \n "
    "以下是检索到的⽂档： \n\n {context} \n\n"
    "以下是⽤⼾的问题： {question} \n"
    "如果⽂档包含与⽤⼾问题相关的关键字或语义，则将其评为相关。 \n"
    "给出⼀个⼆元分数“yes”或“no”，以表明该⽂档是否与问题相关。"
)
def similar_check(state: MessagesState):
    message = state["messages"]
    context = message[-1].content
    question = message[0].content
    prompt = GRADE_PROMPT.format(context=context, question=question)
    response = model.with_structured_output(similar_ret).invoke(prompt)
    ret = response.ret
    if ret == 'yes':
        return "answer_node"
    else:
        return "rewrite_node"
agent_graph.add_conditional_edges(
    "tool_node",
    similar_check,
)
agent_graph.add_edge("rewrite_node", "generate_resp_or_invoke_tool")
agent_graph.add_edge("answer_node", END)

agent = agent_graph.compile()

if __name__ == "__main__":
    query = "比特C++开发方向都有哪些课程"
    final_ans = ""
    for chunk in agent.stream(
            {
                "messages": [HumanMessage(content=query)]
            }
    ):
        for node, update in chunk.items():
            # 只截取 answer_node 的最终内容
            if node == "answer_node":
                final_ans = update["messages"][-1].content

    # 只打印最终结果
    print(final_ans)