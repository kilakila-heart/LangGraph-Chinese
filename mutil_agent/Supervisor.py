import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph_supervisor import create_supervisor

# 初始化 LLM
load_dotenv()

# --- 初始化模型，并绑定工具 ---
os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
model = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的聊天模型
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从DeepSeek获取的API key
    openai_api_base="https://api.deepseek.com/v1"  # DeepSeek的Base URL
)


# 工作 agent A：research 专家（假设支持一个 web_search 工具）
def web_search(query: str) -> str:
    """
    一个用于搜索的API
    """
    # 在这里你可以接入真正搜索 API；为了示例，我们假设返回一个固定的字符串
    # 或者你可以用 LangChain 的某个 search tool
    return f"搜索结果（模拟）: 关于 “{query}” 的信息是 …"


research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_agent",
    prompt="你是研究专家，不做数学运算。如果收到数学运算问题，返回“我不负责数学运算”"
)


# 工作 agent B：数学专家
def add(a: float, b: float) -> float:
    """
    add two numbers return sum
    """
    return a + b


def multiply(a: float, b: float) -> float:
    """
    multiply two numbers
    """
    return a * b


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_agent",
    prompt="你是数学专家，只做数学运算。如果输入不是数学问题，返回“我不负责检索”"
)


# Supervisor 节点，带 state 追踪哪个 agent 最近被调用
# 创建 supervisor，并设置 output_mode="last_message" 避免 full history 出现多个 agent 的内容
workflow = create_supervisor(
    agents=[research_agent, math_agent],
    model=model,
    prompt=(
        "你是 supervisor 来管理两个 agent：research_agent 和 math_agent。\n"
        "如果是检索／事实类问题给 research_agent；"
        "如果是数学运算则给 math_agent。\n"
        "你自己不做具体任务，只负责分配，并且每次只调用一个 agent。"
    ),
    output_mode="last_message",  # 关键：只保留被选中的 agent 的最后输出
    parallel_tool_calls=False     # 确保不要同时并行调用多个 agent
)
app = workflow.compile()

# 测试调用
result_state = app.invoke({
    "messages": [
        {"role":"user", "content":"北京的天气怎么样？"}
    ]
})

# 假设 result_state["messages"] 是一个 List[BaseMessage]，每个元素是 HumanMessage 或 AIMessage

for m in result_state["messages"]:
    # 判断 m 是哪种 message 类型，以确定角色
    if isinstance(m, HumanMessage):
        role = "user"
    elif isinstance(m, AIMessage):
        role = "assistant"
    else:
        # 如果还有 ToolMessage / SystemMessage / 其他
        role = m.__class__.__name__  # 或者一个默认角色

    # 用属性 m.content 获取内容
    content = m.content

    print(f"{role}: {content}")