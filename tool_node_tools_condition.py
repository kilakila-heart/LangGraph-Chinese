import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """返回 a * b"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """返回 a + b"""
    return a + b

tools_list = [multiply, add]

# 2. 准备支持工具调用的 LLM
# llm = ChatOpenAI(model="gpt-4")  # 或者你用别的模型，只要支持 tool calling
# --- 初始化模型，并绑定工具 ---
load_dotenv()
os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
llm = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的聊天模型
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从DeepSeek获取的API key
    openai_api_base="https://api.deepseek.com/v1"  # DeepSeek的Base URL
)


llm_with_tools = llm.bind_tools(tools_list)

# 3. 定义 State，至少包含 messages 用来追踪对话历史
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # 如果你还想在 state 中存别的信息，也可以定义

# 4. 定义节点函数
def chatbot_node(state: State) -> dict:
    """LLM 接收 state['messages']，产生一个新消息（可能包含 tool_calls）"""
    # .invoke 返回 AIMessage(s)，放到 messages 中
    new = llm_with_tools.invoke(state["messages"])
    return {"messages": [new]}

tool_node = ToolNode(tools=tools_list)

# 5. 构造图
builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", tool_node)

# 6. 连边（edges）
builder.add_edge(START, "chatbot")

# 条件边：如果 LLM 要调用工具 → 去 tools；否则 → 结束/返回
builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # 预置条件
    {
        "tools": "tools",     # tools_condition 返回 "tools" 表示要调用工具，这里映射到我们定义的工具节点名 "tools"
        "__end__": "__end__",   # 表示不调用工具，就结束（END 节点）
    }
)

# 工具调用完了，再回到 chatbot 或做其他处理
builder.add_edge("tools", "chatbot")

# 最后编译 graph
graph = builder.compile()

# 7. 使用
# graph.invoke / graph.stream 等方式启动对话
# 初始化 state
initial_state = {"messages": [{"role": "user", "content": "12*(1+30)=？"}]}

# 调用 invoke
result_state = graph.invoke(initial_state)
print(result_state["messages"])

