from typing import TypedDict
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

# 定义状态结构
class MyState(TypedDict):
    user_prompt: str
    generated_text: str

def generate_node(state: MyState) -> MyState:
    # 假设这里调用语言模型生成了一段文本
    # 但在 “工具调用” 或 “生成结果” 之后，我们暂停让人类审核
    gen = f"这是基于提示 '{state['user_prompt']}' 生成的初稿。"
    # 触发 interrupt，让人类编辑这段生成的文本
    edited = interrupt({
        "generated_text": gen
    })
    # resume 后，用人类编辑的内容替代
    return {
        "user_prompt": state["user_prompt"],
        "generated_text": edited
    }

def final_node(state: MyState) -> MyState:
    # 最终做一些处理，例如发布 / 存入数据库 / 输出
    print("最终生成文本为：", state["generated_text"])
    return state

def build_graph():
    builder = StateGraph(MyState)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", final_node)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "finalize")
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph

def run():
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # 运行至 interrupt
    result = graph.invoke({"user_prompt": "写一首诗", "generated_text": ""}, config=config)
    # result 中含有 __interrupt__ 字段，描述挂起时人类需要做的事情
    print("中断信息：", result.get("__interrupt__"))
    # 假设人类看完后，编辑了文本：
    human_edit = input("请输入新的诗句")
    # 恢复执行
    resumed = graph.invoke(Command(resume=human_edit), config=config)
    print("恢复后的状态：", resumed)

if __name__ == "__main__":
    run()
