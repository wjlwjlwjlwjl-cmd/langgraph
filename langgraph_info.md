# 一、概念介绍

## 1.1 Agent 和 WorkFlow

> An agent is AI-powered software that accomplished a goal. 

**Agent**，是实现目标的人工智能软件，能够感知环境输入、自主决策、规划行动路径，核心在于：**由大语言模型（LLM）动态控制流程走向**

**工作流**，是将一个复杂的过程，分解为定义明确、顺序执行的任务流程。如果我们将 **Agent比作聪明的执行者**，那么 **工作流就是我们为它设计的行动蓝图**，一言以蔽之：**预设的、可重复的流程路径**

**LangGraph** 实现的就是工作流

**Agent 和 WorkFlow 的区别：**

 1. Agent 由大模型控制流程走向，灵活性高；WorkFlow 流程固定，步骤预设
 2. Agent 适合开放任务求解，比如科研推理；WorkFlow 适合执行流程明确的任务，比如客服服务

## 1.2 LangGraph 的意义

**Agent Server**，即智能服务。我们在 LangChain 中使用的 LLM，直接作为 Agent Server 的话，会面临**状态丢失**、**难以调试**、**无法干预**、**部署困难**等等问题

LangGraph 是一个强大且灵活的 **“Agent Server 操作系统内核”**，不关心我们用的是什么样的模型、提示词，而是为我们解决构造复杂、可靠、可交互的 Agent Server 时所面临的**状态管理**、**流程编排**、**持久化**、**人工监督**等底层工程难题

# 二、入门：构建 AI 工作流

## 2.1 【案例一】物流系统

### 2.1.1 定义状态

**State** 就像快递上的信息单，记录着人物的各种信息，每一个节点都可以获取、修改，在这个工作流的过程中持续存在，并且有明确的字段定义，即：**共享化、持久化、结构化**

定义 State，可以使用 `pydantic`，但是使用 `TypedDict` 要更加轻量快捷

```python
class package_state(TypedDict):
    package_id: str
    origin: str
    destination: str
    status: str
    history: Annotated[List[str], add] 
    total_distance: Annotated[int, add]
    priority: str
```

对于直接指定了类型的字段，更新策略都是**覆盖更新**，即：只要节点返回了字段的新值，就设置为新的值；对于 `histroy` 和 `reducer` ，我们指定了 Reducer 函数，即如何更新的策略：对于 `history` 来说，采取追加更新的方式，对于 `total_distance` ，采取累加的方式（即不同类型对 `operator.add` 的处理方式不同）

### 2.1.2 构建 StateGraph 图

#### 2.1.2.1 定义节点

每个节点都完成单一的事件处理。所谓节点，其实就是一个**函数**。

每个节点先接收状态，然后返回状态更新；节点之间不直接通信，而是通过 State 交互

```python
def receive_node(state: package_state):
    return {
        "status": "已揽收",
        "history": ["快件揽收完成"]
    }
def sort_node(state: package_state):
    destination = state["destination"]
    if destination == "北京":
        new_dest = "北京"
    elif destination == "上海":
        new_dest = "上海"
    else:
        new_dest = "其他接收地"
    return {
        "status": f"在已{new_dest}分拣",
        "history": [f"快件在{new_dest}分拣完成"]
    }
def standard_transmit(state: package_state):
    return {
        "status": "快件经由标准线路运输中",
        "history": ["标准线路运输"],
        "total_distance": 800
    }
def compress_transmit(state: package_state):
    return {
        "status": "快件经由加急线路运输中",
        "history": ["加急线路运输"],
        "total_distance": 500
    }
def delivery_node(state: package_state):
    return {
        "status": "已送达",
        "history": ["快件配送完成"]
    }
```

#### 2.1.2.2 定义图与添加节点

直接使用我们的状态即可构建图，使用 `add_node` 添加我们上面的节点，同时指定节点名称

```python
package_graph = StateGraph(package_state)

package_graph.add_node("揽收站",receive_node)
package_graph.add_node("分拣站", sort_node)
package_graph.add_node("派送站", delivery_node)
package_graph.add_node("标准运输线路", standard_transmit)
package_graph.add_node("加急运输线路", compress_transmit)
```

#### 2.1.2.3 连接节点

现在我们只是让图中有了节点，但是连接起来才能够构建出一个工作流。`Edge` 分为几种类型：起止边、普通边、条件边。

对于起止边和普通边，直接使用 `add_edge(start_node, end_node)` 即可，使用 `langgraph.graph` 下的 `START, END` 即可

对于条件边，需要使用 `add_conditional_edge`，其中需要指定**起始节点**、**path（确定什么情况走到什么节点**、**path_map（如果path返回的是节点名称，那么不需要这个字段；如果path返回的是节点别名，那么需要它完成节点别名到节点的映射**

```python
def select_transmit(state:package_state):
    if state["priority"] == "加急":
        return  "加急运输线路"
    elif state["priority"] == "标准":
        return "标准运输线路"
package_graph.add_edge(START, "揽收站")
package_graph.add_edge("揽收站", "分拣站")
package_graph.add_conditional_edges(
    "分拣站",
    select_transmit,
)
package_graph.add_edge("标准运输线路", "派送站")
package_graph.add_edge("加急运输线路", "派送站")
package_graph.add_edge("派送站", END)
```

#### 2.1.2.4 编译图

C++ 的编译是完成从高级语言到与平台绑定的机器码，Java 的编译是完成从高级语言到平台无关的低级语言——字节码，而图的编译发生在运行时：配置连接与检查连接正确性（连通性、是否有孤立节点、是否成环）

```python
delivery_system = package_graph.compile()
```

#### 2.1.2.5  运行图

编译之后的图实现了 Runnable 接口，支持异步调用、流式传输、批处理、运行

```python
test = {
        "package_id": "pid2",
        "origin": "北京",
        "destination": "临沂",
        "status": "",
        "history": [],
        "total_distance": 0,
        "priority": "加急"
}
result = delivery_system.invoke(test)
```


## 2.2 【案例二】基于 LangGraph 的代理式 RAG 系统

在这里，所谓代理式，就是指的有自主决策能力是否调用工具、是否产出符合要求的结果的智能逻辑链路

```python
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
docs = ["cpp.md", "info.md", "java.md", "test.md"]  
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
    description="当回答和编程语言有关的信息时，用来检索信息"  
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
    "{question}"    "\n ------- \n"  
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
    "你是负责回答问题的助⼿。 "    "使⽤以下检索到的上下⽂⽚段来回答问题。 "    "如果你不知道答案，就说你不知道。 "    "最多只⽤三句话，回答要简明扼要。\n"  
    "Question: {question} \n"  
    "Context: {context}")  
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
    "你是⼀个评分员，评估检索到的⽂档与⽤⼾问题的相关性。 \n "    "以下是检索到的⽂档： \n\n {context} \n\n"  
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
```

* ToolNode 直接构造工具节点

```python
from langgraph.prebuilt import ToolNode, tools_condition
retriever = vector_store.as_retriever()  
retriever_tool = create_retriever_tool(  
    retriever=retriever,  
    name="文档检索工具",  
    description="当回答和比特有关的信息时，用来检索信息"  
)
tool_node = ToolNode([retriever_tool])
```

* `tool_condition` 判断条件边是否需要去调用工具节点

```python
from langgraph.prebuilt import tools_condition
rag_agent.add_conditional_edges(  
    "generate_query_or_response",  
    tools_condition,  
    {  
        "tools": "tool_node",  
        "__end__": END  
    }  
)
```

这里就看到 `path_map` 的作用了：tools_condition 会根据上一个节点返回的内容中是否含有 `tool_calls` ，来判断是否需要调用工具节点。因为并不知道下一个工具节点的名称，所以使用 `tools` 代替，不需要就返回 `__end__` 表示结束调用。具体哪种返回结果调用哪个节点，就取决于我们 `path_map` 的映射了

## 2.3 LangGraph 其他特性

### 2.3.1 使用 Overwrite 替代 Reducer

前面我们讲过，可以使用 Reducer 指定状态的更新逻辑，但是有的时候，就是需要在某些条件下叠加状态、其他条件下替代状态

```python
class State(TypedDict):  
    messages: Annotated[List[str], operator.add]
    
def replace_message(state: State):  
    return {  
        "messages": Overwrite(["replacement message"])  
    }
```

这样，直接更新就是追加，使用 `Overwrite` 就是替换整个消息列表

### 2.3.2 自定义输入输出格式

一开始我们就提到过，状态是在整个工作流中每个节点都可见、可修改的，这意味着其中包含着输入内容、输出内容、中间内容。如果直接一股脑返回的化，会导致杂乱无章以及客户端额外的工作。

LangGraph 允许我们通过 `StateGraph` 的 `input_schema` 和 `output_schema` 自定义输入输出格式。

自定义输入输出格式，在以下领域往往比较重要：

* API 接口开发：有明确的请求、应答格式
* 微服务：服务间有明确的的数据契约
* 数据管道：明确的输入输出规范

```python
class InputSchema(TypedDict):  
    question: Annotated[str, Field(description="问题描述")]  
  
class OutputSchema(TypedDict):  
    answer: Annotated[str, Field(description="答案")]  
  
class ProcessState(InputSchema, OutputSchema):  
    pass  # Python 中的空语句，占位作用
  
agent_graph = StateGraph(  
    ProcessState,  
    input_schema=InputSchema,  
    output_schema=OutputSchema  
)
```

### 2.3.3 在节点间传递私有数据

有的时候，我们会给 LLM 用户的、自己的私有数据，作为参考。这时为了隐私性，我们不应该让必须使用这些数据的节点之外的节点获得这些数据。这是我们就可以使用私有状态来传递私有数据

私有数据在下面的情况下常用：

* 数据处理的中间结果
* 认证流程， 如令牌的认证流程
* 错误处理，把错误限制在工作流内部，只对外界提供友好信息

```python
class Node1Output(TypedDict):  
    sensitive_data: str = Field(description="私密数据")  
class Node2Input(TypedDict):  
    sensitive_data: str = Field(description="私密数据输入")  
class OverallState(TypedDict):  
    data: str  
  
def node1(state: OverallState) -> Node1Output:  
    return {  
        "sensitive_data": "敏感数据"  
    }  
def node2(state: Node2Input) -> OverallState:  
    return {  
         "data": "经过处理后的敏感数据"  
    }  
def final_output(state: OverallState):  
    return {  
        "data":  f"最后结果：{state["data"]}"  
    }  
```

上面就模拟了私有数据的传递过程：node1 将需要的私有数据给 node2，node2 再使用私有数据生成中间数据，并不把这些私有内容交给下流节点

# 三、工作流模式

## 3.1 提示链模式（Prompt Chaining）

这个工作流成一个流水线，每一步的输出都是上一步的输入。

比如写文章的例子：`生成大纲` -> `生成初稿` -> `润色初稿` -> `最后结果`

```python
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
```

得到的结果：

```text
**文章终稿：**
**一、童真的深渊**  
纯真在黑暗中悄然消逝，如同晨曦被夜幕吞噬。曾经明亮的眼神，如今变得空洞而无神，仿佛失去了灵魂的光芒。那是被剥夺的童年，是无法挽回的伤痛。每一个孩子本应在阳光下奔跑、欢笑，却在无形的枷锁中被迫成长，过早地承受不属于他们的沉重。童年的消逝，不仅是一个个体的悲剧，更是整个社会道德底线的崩塌。
**二、罪恶的交易**  
金钱与欲望交织成一张扭曲的网，将无辜者卷入其中，成为交易的筹码。在这场没有赢家的游戏中，道德被践踏，人性被腐蚀。每一次交易的背后，都是一次对生命的漠视与对正义的背叛。这不仅是一场利益的角逐，更是一场关于良知与底线的较量。当善良沦为牺牲品，当纯洁成为交易的对象，我们不得不反思：究竟是谁在推动这场悲剧？又该由谁来为这一切负责？
```

## 3.2 并行模式（Parallelization）

通过协程的方式，实现用户级的并行处理多个任务，适合有多个彼此互不依赖的任务需要处理的情况

```python
agent_graph = StateGraph(State)  
agent_graph.add_node(tech_node)  
agent_graph.add_node(competitor_node)  
agent_graph.add_node(user_node)  
  
agent_graph.add_edge(START, "tech_node")  
agent_graph.add_edge(START, "competitor_node")  
agent_graph.add_edge(START, "user_node")
```

## 3.3 路由模式（Routing）

路由模式也被称为智能分流，能够根据输入内容自行判断工作流走向，最常用的就是 <u>客服系统</u>

**路由模式核心在于，条件路由的设计**

```python
class Route(TypedDict):  
    step: Literal["before_sale", "after_sale", "technical"] = Field("根据请求类型选择相应的处理方式：售前、售后、技术问题")  
    
def model_route(state: State):   # 通过结构化输出，由大模型决定路由走向
    step = model.with_structured_output(Route).invoke(state["input"])["step"]  
    return {  
        "step": step  
    }   
  
def route_function(state: State):   # 将大模型的路由选择映射到工作流节点
    if state["step"] == "before_sale":  
        return "before_sale_node"  
    elif state["step"] == "after_sale":  
        return "after_sale_node"  
    else:  
        return "technical_node"
```

## 3.4 协调者-工作者模式（Orchestration-Worker）

**协调者-工作者模式**与**并行模式**相比，协调者-工作者模式的任务是由 LLM 自动划分的，而并行模式的任务是在设计的时候就固定下来的。

**协调者-工作者模式的核心在于任务分配模式**。首先由协调者自动将任务划分成几个部分，随后可以使用 LangGraph 为支持这种工作流模式而提供的 Send 将状态传递给工作者，最后由合成者将所有工作者的内容合成在一起

```python
class Section(TypedDict):  
    name: str  
    description: str  
class Sections(TypedDict):  
    sections: List[Section]  
class State(TypedDict):  
    topic: str  
    sections: List[Section]  
    completed_sections: Annotated[List[str], operator.add]  
    result: str  
   
def orchestrator(state: State):  
    planner = model.with_structured_output(Sections)  
    sections = planner.invoke(f"请你做一篇关于{state["topic"]}的报告大纲，一共三个章节")  
    return {  
        "sections": sections["sections"]  
    }
def mission_assigner(state: State):  
    sections = state["sections"]  
    workers = []  
    for section in sections:  
        workers.append(Send("llm_node", {"section": section}))  
    return workers   
def llm_node(state: State):  
    section = state["section"]  
    name = section["name"]  
    description = section["description"]  
    result = model.invoke(f"编写报告章节：{name}, 编写内容要求：{description}").content  
    return {  
        "completed_sections": [result]  
    } 
def synthesizer(state: State):  
    content = "\n\n" .join(state["completed_sections"])  
    result = f"# {state['topic']}\n\n{content}"  
    return {  
        "result": result  
    }
```

## 3.5 评估器-优化器模式

前面我们实现的 【案例二】基于 LangGraph 的代理式 RAG 系统 就是评估器、优化器模式，核心在于**质量检测机制**。评估器决定生成结果是否合格，不合格就交给优化器优化问题，重新生成