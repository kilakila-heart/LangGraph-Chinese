import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.types import Command

from util import get_langgraph_display

# 用于画图
os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111"

# 初始化 LLM
load_dotenv()

# --- 初始化模型，并绑定工具 ---
os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
llm = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的聊天模型
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从DeepSeek获取的API key
    openai_api_base="https://api.deepseek.com/v1"  # DeepSeek的Base URL
)


# 定义两个 agent 的行为函数
def agent_A(state: MessagesState) -> Command[Literal["agent_B", END]]:
    msgs = state["messages"]
    last = msgs[-1].content if msgs else ""
    if "数学" in last:
        # 转到 agent_B
        return Command(goto="agent_B", update={})
    else:
        resp = llm.invoke(msgs + [("assistant", "Agent A 回答: “非数学问题已处理”")])
        # 更新 state.messages
        return Command(
            goto=END,
            update={"messages": state["messages"] + [{"role": "assistant", "content": resp.content}]}
        )


def agent_B(state: MessagesState) -> Command[Literal["agent_A", END]]:
    msgs = state["messages"]
    resp = llm.invoke(msgs + [("assistant", "Agent B 回答: “这是数学问题的解答部分”")])
    return Command(
        goto="agent_A",
        update={"messages": state["messages"] + [{"role": "assistant", "content": resp.content}]}
    )


# 构建 StateGraph
graph_builder = StateGraph(MessagesState)

# 添加节点
graph_builder.add_node("agent_A", agent_A)
graph_builder.add_node("agent_B", agent_B)

# 设置入口
# 方法一: set entry point
graph_builder.set_entry_point("agent_A")

# 添加边（控制流）
graph_builder.add_edge("agent_A", "agent_B")
graph_builder.add_edge("agent_A", END)
graph_builder.add_edge("agent_B", "agent_A")

graph_builder.add_edge("agent_B", END)
# 编译 graph
compiled_graph = graph_builder.compile()
get_langgraph_display(compiled_graph, "network.py")

# 初始化 state
initial_state = {"messages": [{"role": "user", "content": "你能帮我解一个数学问题吗？"}]}

# 调用 invoke
result_state = compiled_graph.invoke(initial_state)

print("结束后的对话历史:")
for msg in result_state["messages"]:
    print(f'{msg["role"]}: {msg["content"]}')
