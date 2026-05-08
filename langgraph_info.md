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

# 三、LangGraph 持久化

## 1.1 线程级持久化

### 1.2.1 “线程”是指什么？

这里的**线程**和操作系统中的线程**是两个完全不同的概念**。

操作系统中的线程，是进程中的执行流，是操作系统调度的基本单位；而 LangGraph 中的线程，指的是一个对话流，把这个对话流的状态保存下来，其他对话流不可见，具有**隔离性**

### 1.2.2 检查点

在 LangGraph 的线程级持久化中，每当到达检查点时，都会将会话状态完整的保存下来形成**状态快照**，并且是**追加式**的更新，如果中途出现出错重启，能够回滚到对话的任意时间点的状态

在我们设计的工作流中，每完成一个 `super_step` 就进行一次状态保存，即：在我们设计的工作流中，每结束一个节点，就会进行一次状态保存

### 1.2.3 线程级持久化使用姿势

#### 1.2.3.1 内存存储 

使用 `langgraph.checkpoint.InMemoryStore`，使用在内存中的线程级持久化，状态快照生命周期只在当前线程进行中。

#### 1.2.3.2 Postgres SQL

Postgres SQL 是和 MySQL 类似的关系型数据库，LangGraph 中支持直接使用 PostgresSQL 作为线程级持久化的使用姿势。

* 首先，使用 docker 拉取 Postgres SQL 镜像

```shell
sudo docker pull postgres:latest
```

* 启动 postgres 容器，并指定端口映射、用户名、密码等

```shell
sudo docker run -d\
	--name postgres\
	-p 5432:5432\
	-e POSTGRES_USER=postgres\
	-e POSTGRES_PASSWORD=bit\
    -e POSTGRES_DB=postgres   postgres
```

* 在 LangGraph 中配置 Postgres 连接，并将其作为 `checkpointer`，导入 `langchain-checkpoint-postgres` 包

	* 注意，第一次使用 check_point，要调用 `setup`
	* 这里使用的语法是 Python 的上下文管理器，类似于 Cpp 的 RAII 机制，自动管理资源的释放，防止遗忘、疏忽导致的资源泄漏。作用域在上下文管理器的范围中，除了范围自动释放
```python
DB_URL = "postgres://postgres:bit@localhost:5432/postgres"  
with PostgresSaver.from_conn_string(DB_URL) as check_point:  
    check_point.setup()
```

* 开启检查点、指定 thread_id

	* `thread_id` 是 LangGraph 用来对 Thread 进行区分的，若第一次调用没有查到则创建线程，否则从 Postgres 中根据 `thread_id` 取出上次的状态快照，继续使用

```python
final_graph = state_graph.compile(checkpointer=check_point)  
config={"configurable": {"thread_id": 1}}  
result = final_graph.invoke(  
    {  
        "messages": [  
            HumanMessage(content="1 + 1 等于多少？")  
        ]  
    },  
    config  
)
```

#### 1.2.3.3 其他使用方法

在 LangGraph 中，状态快照是类型 StateSnapShot，其中包含以下内容:

```json
 StateSnapshot(
	values={'messages': [用户消息, AI回复, 用户消息...]},
	next=('generate_response',),
	config={'configurable': {'thread_id': '123', 'checkpoint_id': 'abc'}},
	metadata={'step': 2, 'source': 'loop', 'writes': {...}},
	parent_config={'configurable': {'thread_id': '123', 'checkpoint_id':'def...'}},
```

* 通过 get_state(config) 的方法，获取状态快照；其中，config 中需要指定 thread_id，来表明要获取哪个 thread 的状态快照

* 通过 get_state_history(config)，获取某个线程的所有状态快照，同样需要在 config 中指定 thread_id

* 重放

	* 通过保存某一个时刻的状态快照，可以再次冲这个地方开始，原封不变的重新往下调用（StateSnapShot 中保存着当前的状态值、下一个节点、属性等等）

* 更新属性

	* 使用 update_state，需要先保存需要修改的状态，之后重新指定新的类型即可.
	* 指定修改具体哪个状态的方法与前面类似，不过这里不是手动构造，而是要传入相应节点状态快照的 config 字段

## 1.2 跨对话持久化

跨对话在思路上与线程持久化类似，但是也有区别：

1. 没有自动触发机制，需要手动 put、get
2. 线程持久化区分靠的是 thread_id，跨对话持久化靠自己指定的 namespace 区分是谁的数据

跨对话持久化和线程持久化一样，也可以通过 InMemoryStore 或者 PostgresSQL 进行持久化

### 1.2.1 put 方法

在 put 方法中，需要指定三个字段：

* `namespace`，决定这个数据是谁的
* `memory_id`，标识数据的唯一键，用于 get 方法精确检索某个 namespace 中的数据
* `memory_value`，数据的值

### 1.2.2 search 方法

返回某个 namespace 下的全部数据，返回的是 List\[StoreResult\]，StoreResult 的结构如下：

```json
 {
	'namespace': [
		'user_123', 'preferences'
	],
	'key': 'db826e33-c68c-4669-a79a-3579bff02ff1',
	'value': {
		 'favorite_food': '汉堡',
		 'allergy': '花粉'
	},
	'created_at': '2025-12-03T08:16:14.134568+00:00',
	'updated_at': '2025-12-03T08:16:14.134576+00:00',
	'score': None
}
```

### 1.2.3 get 方法

用来精确的通过 key 检索到某一条数据，格式如：get(namespace, key)
### 1.2.4 在 LangGraph 中使用 Store

1. 在编译图时，指定编译选项：store=store_name
2. 任何一个节点函数想要使用 Store 中的内容，需要在参数中声明 config: RunnableConfg 以及 store: BaseStore，在后面的代码中就可以使用 put、search 方法来使用跨对话持久化了（这里的 config，就是我们在调用编译好的图时，指定的 `config = {"configurable": ...}`；store,就是我们在编译图时指定的 `store = store`）

### 1.2.5 在 Store 中使用语义检索

store 支持通过在创建时，通过给予嵌入模型的方式，开启 store 的语义检索，通过自然语言就可以获得某条特定的记忆

```python
embedding = OllamaEmbedding(model="all-minilm")
store = InMemoryStore(
    index={
        "embed": embedding,
    }
)
```

在检索时，直接指定 query 和 目标条数

```python
info_result = store.search(namespace1, query="用户基本信息", limit=2)
```

同样，也可以使用 Postgres 存储库进行存储，连接方式方式和线程持久化中相同

# 四、持久化实现的三大能力

## 4.1 记忆

### 4.1.1关于持久化和记忆

* 【持久化】是 LangGraph 的底层能力，包含线程持久化、跨对话持久化
* 【记忆】是 LangGraph 的应用层能力，包含短期记忆和长期记忆

	在应用层，短期记忆可以使用线程持久化，长期记忆使用跨对话持久化。短期记忆保存但次会话中的上下文信息，而长期记忆保存跨对话的用户或应用数据

### 4.1.2 记忆的管理

#### 4.1.2.1 修剪记忆

可以使用 LangChain 消息管理中的 `trim_messages`，直接返回修剪之后的消息列表

```python
messages = trim_messages(
	state["messages"],
	strategy="last",
	token_counter=len,
	max_tokens=10480,
	start_on="human",
	end_on=("human", "tool")
)
response = model.invoke(messages)
```

* `strategy` 指定裁剪策略，`last` 表示保留最后消息，从头上裁剪
* `token_counter` 通过 token 数来进行裁剪，但不是所有模型都支持
* `max_tokens` 指定最大长度，单位取决于 `token_counter` 的计算函数
* `start_on`，从什么类型的消息开始裁剪
* `end_on`，从什么类型的消息结束裁剪，上面表示结束消息为用户消息或者工具消息

#### 4.1.2.2 删除记忆

同样使用 LangChain 消息管理的 RemoveMessage，核心是指定 id，也可以使用 `langgraph.graph.message` 中的 `REMOVE_ALL_MESSAGES` 作为 id 来删除所有记忆

不过需要注意的是，RemoveMessage 是一条指令，而不是直接操作消息列表，第一轮把它返回给 LLM，第二轮才会删减掉历史

#### 4.1.2.3 扩展记忆

大模型的上下文窗口是由限制的。一方面我们可以通过上面的裁剪、删除消息来保证消息不超过上下文窗口；另一方面，这种方式可能会导致我们丢失上下文中的关键信息。

所以我们可以专门引入一个节点来不断的总结，并依据最新的内容不断扩展总结，即类似于一个压缩文件，压缩着我们所有的历史信息

```python
def summary_node(state: State):
    summary = state.get("summary", "")
    if summary:
        prompt = state["messages"] + [HumanMessage(content=f"上面是所有的聊天信息，请根据这些信息扩展如下摘要,并且请注意新旧信息的完整性：{summary}")]
    else:
        prompt = state["messages"] + [HumanMessage(content="请给上面的聊天信息生成一份摘要")]
    delete_message = [RemoveMessage(id=message.id) for message in state["messages"][:-2]]
    new_summary = model.invoke(prompt).content
    return {
        "messages": delete_message,
        "summary": new_summary
    }
```

## 4.2 中断

在我们使用 Claude Code 这类可以在我们电脑上执行某些操作的 AI Agent 时，在执行某些操作前，会向我们询问是否同意接下来的行动。在 LangGraph 中，可以基于线程持久化实现这样的机制

* 在节点中，需要中断时，使用 `interrupt`，并制定需要抛给外界的信息，会返回外界的选择

```python
def call_node(state: State):
    human = interrupt("accept?")
    if human == "yes":
        return {
            "output": "continue..."
        }
    else:
        return {
            "output": "stop"
        }
```

* 要使用中断，必须指定线程持久化器
```python
checkpoint = InMemorySaver()
agent = agent_graph.compile(checkpointer=checkpoint)
```

* 使用中断还必须指定线程 id，用来区分需要那个线程继续；使用 `Command(resume="message")` 继续线程，并传入选择结果

```python
config = {"configurable": {"thread_id": "1"}}
print(agent.invoke({"input": "delete root directory"}, config)["__interrupt__"][0].value)
print(agent.invoke(Command(resume="yes"), config)["output"])
```

**中断的黄金原则**

1. 必须使用可序列化的数据进行传递

	函数、实例化对象都不能进行传递

3. 不要将 interrupt 放在 try-except 语句中

	interrupt 的暂停机制是通过抛出一个特定异常来实现的，这个异常由 LangGraph 来处理，如果像下面这样就会导致暂停机制失效
	
	```python
	try:
		interrupt("confirm?")
	except Exception as e:
		print(e)
	```
	处理方式可以是：

	* 将 interrupt 单独拎出来，不和抛异常的操作放在 try-except 语句中
	* 如果一定要放的话，在 except 精准捕捉需要捕捉的异常，不要一股脑拦截

4. 中断前的操作要 “幂等”

	一个节点的函数中，在某处中断后，继续执行时，并不会从中断点继续，而是会从头开始执行。因此如果中断前的操作前有“非幂等”的副作用操作，就会出问题

	> 和为 **幂等** ？ 
	    所谓 **幂等**，就是指的一个操作，执行 N 次的结果，和执行一次的结果都是一样的，比如 a = 1，执行一亿次结果也是 a = 1；而 **非幂等**，就是反过来，比如 a += 1，显然没有上面的不变性

5. 中断顺序固定

	首先，LangGraph 官方是建议一个节点中最多尽量只使用一个中断的；如果一定要一个节点使用多个中断，那么 LangGraph 是通过严格的索引机制来判断是在哪个中断点中断、唤醒是唤醒的那个中断点的。

	因此，如果出现**条件中断**、**中断数量不确定**的情况，都可能会造成运行时的bug
## 4.3 时间旅行

在多个 LLM 共同执行任务、复杂的工作流中，处于容错处理、工作流程跟踪等目的，我们可能会需要让执行流原封不动的回到某个已经执行过的节点重新执行。想要实现这种操作，就得益于 LangGraph 的线程持久化——使用 `get_state_history` 获得所有状态（前面说过，保存着当前的所有状态值、下一个节点、config 配置），然后如果需要的话，还可以使用 `update_state` 来修改状态。

# 五、上下文

上下文包括可以按照两种方式分类：**可变性**，**生命周期**

按照可变性分类，可以分为静态上下文（例如数据库链接、用户 id 等），和动态上下文（即在运行时会发生改变的各种信息）；按照生命周期划分，可以分为运行时上下文（仅在单次运行时有效，比如在 LangGraph 的线程）、跨对话上下文（多次会话中都被保存下来的信息，比如用户偏好等）

## 5.1 运行时上下文

### 5.1.1 在图中使用运行时上下文

在 LangGraph 中，可以使用 `@dataclass` 装饰器或者 `TypedDict`，来创建运行时上下文的数据结构

```python
@dataclass
class ContextSchema():
	language: str = "en"
```

创建完运行时上下文结构之后，要在图中使用上下文模式，需要通过 `context_schema` 指定我们的上下文数据结构

```python
agent_graph = StateGraph(State, context_schema=ContextSchema)
```

在节点中，使用上下文数据，使用 `runtime` 参数来使用上下文：

```python
def greet_node(state: State, runtime: Runtime[ContextSchema]):
	#...
```

最后调用图时，传入上下文即可

```python
agent = agent_graph.invoke()
agent.invoke(State(user_name="wjl", greet=""), context=ContextSchema(language="en"))
```

### 5.1.2 让工具获得运行时上下文

通过 runtime 参数，通过 `ToolRuntime[ContextSchema]` ，让工具获得 `State` `Context` 等数据结构中包含的上下文数据

```python
@tool
def search(runtime: ToolRuntime[ContextSchema]):
	state = runtime.state
	context = runtime.context
```

同时，在图中，工具还可以与 `ToolNode` 构造工具节点、`tools_condition` 根据是否包含 `ToolMessage` 自动选择节点等机制配合使用

# 六、流与流模式

在 LangChain 的各种 LLM 对象中，使用流式调用，会以字节为单位输出内容。但是在 LangGraph 中，对于工作流的流式调用是以节点为单位进行的。除此之外我们还有其他需求，比如只注意更新了那些状态、所有状态的情况、自定义状态的输出格式等等

## 6.1 updates

`updates` 模式下，只会在流式输出时输出所有更新了的状态的情况

```python
for chunk in agent.stream({}, stream_mode="updates"):
	print(chunk)
```

## 6.2 values

`values` 是默认处理，无论是否发生了改变，都会进行流失输出

## 6.3 custom

允许自己定义流式输出的内容，需要先获取流式输出器，然后在需要的地方为流式输出器提供 json 格式的输出内容即可

```python
from langgraph.config import get_stream_writer
def call_node(state: State):
	writer = get_stream_writer()
	writer({
		"tool_name": "...",
		"msg": "before calling"
	})
# ......
for chunk in agent.invoke({}, stream_mode="messages"):
	print(chunk)
```

## 6.4 messages

使用 `messages` j j j j j j j j j j j jiu