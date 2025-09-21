import os

# 假设 deepseek 的 LLM 类似于 langchain 的模型，可以 bind_tools
# 这里只做伪代码／假设
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from openai.resources.fine_tuning.jobs import Checkpoints
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

load_dotenv()

# --- 初始化模型，并绑定工具 ---
os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
model = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的聊天模型
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从DeepSeek获取的API key
    openai_api_base="https://api.deepseek.com/v1"  # DeepSeek的Base URL
)
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tools = [TavilySearchResults(max_results=3)]

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

agent = create_react_agent(model, tools, messages_modifier=prompt)



# values模式示例
inputs = {"messages": [("human", "2024年北京半程马拉松的前3名成绩是多少?")]}
for chunk in agent.stream(
        inputs,
        stream_mode="values",
):
    print(chunk["messages"][-1].pretty_print())

# updates模式示例
# for chunk in agent.stream(
#         inputs,
#         stream_mode="updates",
# ):
#     print(chunk)
