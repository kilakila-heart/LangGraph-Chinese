import os
from typing import Literal, Any

# 假设 deepseek 的 LLM 类似于 langchain 的模型，可以 bind_tools
# 这里只做伪代码／假设
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END


# --- 定义工具 ---
@tool
def get_weather(city: str) -> str:
    """返回指定城市的天气概况"""
    # 这里假设是查询某个 API，或模拟
    if city.lower() in ["san francisco", "sf"]:
        return "Weather in SF: 60°F, foggy"
    else:
        return f"Weather in {city}: sunny, 75°F"


@tool
def add(a: int, b: int) -> int:
    """
    return the sum of two numbers
    """
    return a + b


# --- 初始化模型，并绑定工具 ---
os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
model = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的聊天模型
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从DeepSeek获取的API key
    openai_api_base="https://api.deepseek.com/v1"  # DeepSeek的Base URL
)
model_with_tools = model.bind_tools([get_weather, add], parallel_tool_calls=True)

# --- 构造 ToolNode ---
tool_node = ToolNode([get_weather, add], handle_tool_errors=True)


# --- 定义状态图的节点函数 ---

def call_model(state: MessagesState) -> dict[str, Any]:
    """
    输入 state（包含 messages list），调用模型，返回新消息 appended。
    """
    msgs = state["messages"]
    # 模型根据已有 message 决定是否要调用工具
    response = model_with_tools.invoke(msgs)
    # 返回一个新的 messages 列表（这里只返回 response 消息作为新的消息）
    return {"messages": msgs + [response]}


def should_call_tools(state: MessagesState) -> Literal["tools", END]:
    """
    检测 state 中的最后一条消息，是否包含工具调用指令。
    如果有 tool_calls，就进入 "tools" 节点执行工具；否则结束。
    """
    msgs = state["messages"]
    last = msgs[-1]
    # AIMessage 类型的消息，假设有 tool_calls 属性
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    else:
        return END


# --- 构造 StateGraph ---
builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_call_tools, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile()

# --- 用例：用户问天气 + 数学问题 ---
initial_state = {"messages": [HumanMessage(content="What's the weather in SF?")]}
result = graph.invoke(initial_state)
print(result["messages"])

initial_state2 = {"messages": [HumanMessage(content="Compute 5 + 7 for me.")]}
result2 = graph.invoke(initial_state2)
print(result2["messages"])
