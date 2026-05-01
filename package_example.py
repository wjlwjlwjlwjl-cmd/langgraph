from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from operator import add

# 定义状态
class package_state(TypedDict):
    package_id: str
    origin: str
    destination: str

    status: str # 覆盖更新
    history: Annotated[List[str], add] # 追加更新历史
    total_distance: Annotated[int, add] # 追加总路程

    priority: str

# 定义图
package_graph = StateGraph(package_state)

# 构造节点
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

# 添加节点
package_graph.add_node("揽收站",receive_node)
package_graph.add_node("分拣站", sort_node)
package_graph.add_node("派送站", delivery_node)
package_graph.add_node("标准运输线路", standard_transmit)
package_graph.add_node("加急运输线路", compress_transmit)

def select_transmit(state:package_state):
    if state["priority"] == "加急":
        return  "加急运输线路"
    elif state["priority"] == "标准":
        return "标准运输线路"
# 添加起点、终点
package_graph.add_edge(START, "揽收站")
package_graph.add_edge("揽收站", "分拣站")
package_graph.add_conditional_edges(
    "分拣站",
    select_transmit,
    ["加急运输线路", "标准运输线路"]
)
package_graph.add_edge("标准运输线路", "派送站")
package_graph.add_edge("加急运输线路", "派送站")
package_graph.add_edge("派送站", END)

# 编译系统
delivery_system = package_graph.compile()

test_package = [
    {
        "package_id": "pid1",
        "origin": "北京",
        "destination": "上海",
        "status": "",
        "history": [],
        "total_distance": 0,
        "priority": "标准"
    },
    {
        "package_id": "pid2",
        "origin": "北京",
        "destination": "临沂",
        "status": "",
        "history": [],
        "total_distance": 0,
        "priority": "加急"
    }
]
for test in test_package:
    result = delivery_system.invoke(test)
    print(result)
    print("*" * 30)