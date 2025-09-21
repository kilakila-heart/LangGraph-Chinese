# -*- coding: utf-8 -*-
# 示例：一个三步推理流程：plan -> search_tools -> evaluate
# 我们运行一次流程，选取 "search_tools" 之后的 checkpoint，
# 修改 search 结果（模拟不同工具输出），然后从该 checkpoint 继续执行，
# 比较两个分支的最终 evaluate 结果（探索性实验）。
# https://chatgpt.com/c/68cf6ca0-8b8c-8326-90bf-f1770989b806
import os
import uuid

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, NotRequired


# ---------------------------
# 1) 定义 State
# ---------------------------
class State(TypedDict):
    query: str
    plan: NotRequired[str]
    search_results: NotRequired[list]  # 假设工具返回一个列表
    evaluation: NotRequired[str]


# ---------------------------
# 2) 初始化 LLM（示例）
# ---------------------------
# 注意：在真实环境中请用实际的 API key / 模型名（这里与文档风格一致）
from dotenv import load_dotenv

load_dotenv()

# 如果需要直接配置API密钥和基础URL
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


# ---------------------------
# 3) 定义节点函数（每个函数返回要更新回 state 的 dict）
# ---------------------------
def plan_step(state: State):
    # 让 LLM 根据 query 生成计划（简化为 single sentence）
    prompt = f"你要解决的问题是：{state['query']}\n请给出一个 1-2 步的简短行动计划。"
    msg = llm.invoke(prompt)
    return {"plan": msg.content}


def search_tools_step(state: State):
    # 模拟调用外部工具或检索；在真实情况里这里可以是 API 调用
    # 这里默认返回两个“候选事实”
    # 如果想要 time-travel 探索，就在 checkpoint 后修改这个字段
    results = [
        {"id": 1, "fact": "A: 广州最近正在下雨"},
        {"id": 2, "fact": "B: 我不喜欢去室外走，腿脚不舒服"},
    ]
    return {"search_results": results}


def evaluate_step(state: State):
    # 用 LLM 把 plan + search_results 组合成结论/建议
    prompt = (
        f"Plan: {state.get('plan')}\n"
        f"Facts: {state.get('search_results')}\n"
        f"基于上面信息，写出简短结论（1 段话）。"
    )
    msg = llm.invoke(prompt)
    return {"evaluation": msg.content}


# ---------------------------
# 4) 构建 StateGraph
# ---------------------------
workflow = StateGraph(State)
workflow.add_node("plan", plan_step)
workflow.add_node("search_tools", search_tools_step)
workflow.add_node("evaluate", evaluate_step)

workflow.add_edge(START, "plan")
workflow.add_edge("plan", "search_tools")
workflow.add_edge("search_tools", "evaluate")
workflow.add_edge("evaluate", END)

# ---------------------------
# 5) 编译并运行（启用 checkpointer）
# ---------------------------
# 5) 设置 MySQL checkpointer
# ---------------------------
# 用你的 MySQL 参数替换下面的 DB_URI
DB_USER = "root"
DB_PASS = "123456"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "langgraph"
DB_URI = f"mysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# 创建 checkpointer 对象
cm = PyMySQLSaver.from_conn_string(DB_URI)
checkpointer = cm.__enter__()    # 强制进入 context manager，拿到 saver
# 第一次运行前建立必要表结构
checkpointer.setup()
# checkpointer = InMemorySaver()  # 演示用；生产请换 SQLite/Postgres 实现



graph = workflow.compile(checkpointer=checkpointer)

# 每次运行一个新的线程 id（这会在 checkpointer 中保存一系列 checkpoint）
config_run = {"configurable": {"thread_id": str(uuid.uuid4())}}
initial_state = {"query": "在广州2天怎么玩？"}

# 运行（首次执行）
final_state = graph.invoke(initial_state, config_run)
print("首次执行, 评估结果：\n", final_state.get("evaluation"), "\n")

# ---------------------------
# 6) 检索历史 checkpoint 并选择要回溯的位置
# ---------------------------
history = list(graph.get_state_history(config_run))  # 返回按时间逆序（文档指明）
print("历史 checkpoints（最近的在前）：")
for s in history:
    print("next:", s.next, "checkpoint_id:", s.config["configurable"]["checkpoint_id"])
print()

# 假设我们选取在 'search_tools' 之后的 checkpoint（history[1]，示例）
selected_state = history[1]
print("选中 checkpoint next:", selected_state.next, "values:", selected_state.values)

# ---------------------------
# 7) 修改该 checkpoint 的 state（time travel）
# ---------------------------
# 我们把 search_results 改成另一个候选集合，模拟不同外部工具返回
new_values = {"search_results": [
    {"id": 99, "fact": "C: 最近广州天气很好，温度适宜"},
    {"id": 100, "fact": "D: 我比较爱好大自然喜欢去外面走"}
]}
new_config = graph.update_state(selected_state.config, values=new_values)
# update_state 返回新的 config，其中包含新的 checkpoint_id

# ---------------------------
# 8) 从修改后的 checkpoint 继续执行（会产生一个新分支）
# ---------------------------
graph.invoke(None, new_config)  # 从该 checkpoint 恢复并执行后续节点
# 这次执行会写入新的 checkpoint（分支）并返回最终 state（上面 invoke 返回的 state 可打印/捕获）

# 再次获取历史来比较分支（可查看所有 checkpoint；或读取最后的 final state）
all_history = list(graph.get_state_history(new_config))
for s in all_history[:5]:
    print("branch next:", s.next, "id:", s.config["configurable"]["checkpoint_id"], "values:", s.values)
