import os

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

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


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  # (1)!
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]


# 使用 DeepSeek 模型
agent = create_react_agent(
    model=llm,  # 修改为 DeepSeek 模型
    tools=[get_weather],
    checkpointer=checkpointer,  # (1)!
    # A static prompt that never changes
    prompt=prompt
)

# Run the agent
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "上海今天要穿什么？"}]},
    config={"configurable": {"user_name": "John Smith", "thread_id": 42}}
)

print(sf_response)
