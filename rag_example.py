from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.retrievers import RetrieverInput
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AnyMessage
from pydantic import Field, BaseModel

model = ChatOpenAI(
    model = "qwen-turbo"
)
embedding = OllamaEmbeddings(
    model = "all-minilm"
)

# 构建检索工具
docs = ["bite_cpp.md", "bite_info.md", "bite_java.md", "bite_test.md"]
chunks = []
for doc in docs:
    data = TextLoader(doc).load();
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks.append(spliter.split_documents(data))
vector_store = InMemoryVectorStore(
    embedding=embedding
)
vector_store.add_documents([item for sublist in chunks for item in sublist])

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="文档检索工具",
    description="当回答和比特有关的信息时，用来检索信息"
)

# 定义节点

# 决定是否需要调用工具
def generate_query_or_response(state:MessagesState):
    response = (
        model.bind_tools([retriever_tool], tool_choice="文档检索工具").invoke(state["messages"])
    )
    return {
        "messages": [response]
    }


# 工具节点
tool_node = ToolNode([retriever_tool])

# 问题优化节点
REWRITE_PROMPT = (
    "查看输⼊并尝试推断潜在的语义意图/含义。\n"
    "这是最初的问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "提出⼀个改进后的问题："
)
def regenerate_node(state: MessagesState):
    message = state["messages"]
    question = message[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [{
            "role": "user",
            "content": response.content
        }]
    }

# 结果生成节点
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
    content = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([HumanMessage(content=content)])
    return {
        "messages": [response]
    }

# 添加节点
rag_agent = StateGraph(MessagesState)
rag_agent.add_node(generate_query_or_response)
rag_agent.add_node("tool_node", tool_node)
rag_agent.add_node(answer_node)
rag_agent.add_node(regenerate_node)

GRADE_PROMPT = (
    "你是⼀个评分员，评估检索到的⽂档与⽤⼾问题的相关性。 \n "
    "以下是检索到的⽂档： \n\n {context} \n\n"
    "以下是⽤⼾的问题： {question} \n"
    "如果⽂档包含与⽤⼾问题相关的关键字或语义，则将其评为相关。 \n"
    "给出⼀个⼆元分数“yes”或“no”，以表明该⽂档是否与问题相关。"
)
class Relevant(BaseModel):
    ret: str = Field(description="如果有关，返回'yes', 否则返回'no'")
def JudgeRelevant(state:MessagesState):
    message = state["messages"]
    question = message[0].content
    context = message[-1].content
    response = model.with_structured_output(Relevant).invoke([
        HumanMessage(content=GRADE_PROMPT.format(context=context, question=question))
    ])
    ret = response.ret
    if ret == "yes":
        return "answer_node"
    else:
        return "regenerate_node"

rag_agent.add_edge(START, "generate_query_or_response")
rag_agent.add_conditional_edges(
    "generate_query_or_response",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)
rag_agent.add_conditional_edges(
    "tool_node", 
    JudgeRelevant,
)
rag_agent.add_edge("regenerate_node", "generate_query_or_response")
rag_agent.add_edge("answer_node", END)

agent = rag_agent.compile()

while True:
    print("# ", end="")
    question = input()
    for chunk in agent.stream({"messages": [HumanMessage(content=question)]}):
        for node, update in chunk.items():
            if node == "answer_node":
                print(">>> " + update["messages"][-1].content)
                print()
            