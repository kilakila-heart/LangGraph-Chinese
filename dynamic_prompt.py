import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

checkpointer = InMemorySaver()

load_dotenv()


def get_weather(city: str) -> str:  # (1)!
    """Get weather for a given city."""
    return f"It's always rainy in {city}!"


# 如果需要直接配置API密钥和基础URL
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 使用 DeepSeek 模型
agent = create_react_agent(
    model=llm,  # 修改为 DeepSeek 模型
    tools=[get_weather],
    checkpointer=checkpointer,  # (1)!
    # A static prompt that never changes
    prompt="Never answer questions about the weather."
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "上海今天要穿什么？"}]},
    config  # (2)!
)

print(sf_response)
