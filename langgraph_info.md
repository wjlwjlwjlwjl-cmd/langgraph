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

## 2.1 物流系统

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

### 2.1.3 其他工具

#### 2.1.3.1 ToolNode 直接构造工具节点

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

#### 2.1.3.2 `tool_condition` 判断条件边是否需要去调用工具节点

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