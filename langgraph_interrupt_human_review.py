from typing import TypedDict, Optional
import uuid

from langgraph.graph import StateGraph, END
from langgraph.constants import START
from langgraph.checkpoint.memory import InMemorySaver  # 或换成更持久化的
from langgraph.types import interrupt, Command

# 假设有个工具 publish_article
def publish_article_tool(article_id: str, summary: str) -> str:
    """
    模拟工具调用：发布文章，返回发布记录或状态
    """
    # 在现实中可能是 HTTP API 之类的调用
    return f"Article {article_id} published with summary: {summary}"

# 状态结构
class State(TypedDict, total=False):
    article_id: str
    article_text: str
    summary: str
    summary_reviewed: bool
    publish_status: Optional[str]
    user_feedback: Optional[str]

# Node1: 生成摘要
def generate_summary(state: State) -> State:
    text = state["article_text"]
    # 假设调用 LLM 生成摘要
    summary = f"自动生成的摘要（基于文章内容: {text[:50]}...）"
    return {
        "article_id": state["article_id"],
        "article_text": state["article_text"],
        "summary": summary,
        "summary_reviewed": False
    }

# Node2: 人类审核摘要
def review_summary(state: State) -> State:
    # 提供摘要，让人类审查
    edited_summary = interrupt({
        "summary_to_review": state["summary"]
    })
    # resume 时把人类修改的总结放回
    return {
        **state,
        "summary": edited_summary,
        "summary_reviewed": True
    }

# Node3: 决定是否发布
def decide_publish(state: State) -> Command:
    # 假设人类审核后，summary_reviewed=True
    # 再让人类“批准或拒绝发布”
    decision = interrupt({
        "question": "审核过的摘要如下，请决定是否发布：",
        "summary": state["summary"]
    })
    decision = decision.strip().lower()
    if decision == "yes" or decision == "approve":
        return Command(goto="publish")
    else:
        return Command(goto="reject_publish")

# Node4: 工具调用：发布文章
def publish(state: State) -> State:
    pub_status = publish_article_tool(state["article_id"], state["summary"])
    return {
        **state,
        "publish_status": pub_status
    }

# Node5: 发布后确认或反馈
def post_publish_feedback(state: State) -> State:
    feedback = interrupt({
        "feedback_request": "文章已发布，您是否满意？",
        "publish_status": state.get("publish_status", "")
    })
    return {
        **state,
        "user_feedback": feedback
    }

# Node6: 终点：摘要被拒绝发布
def reject_publish(state: State) -> State:
    return {
        **state,
        "publish_status": "rejected_by_human"
    }

# Node7: 终点：发布成功并收集反馈
def done(state: State) -> State:
    # 可以在这里记录 final 日志或存数据库
    print("流程完成。状态：", state)
    return state

# 构建 graph
def build_graph():
    builder = StateGraph(State)
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("review_summary", review_summary)
    builder.add_node("decide_publish", decide_publish)
    builder.add_node("publish", publish)
    builder.add_node("post_publish_feedback", post_publish_feedback)
    builder.add_node("reject_publish", reject_publish)
    builder.add_node("done", done)

    # 定义边／流程
    builder.set_entry_point("generate_summary")
    builder.add_edge("generate_summary", "review_summary")
    builder.add_edge("review_summary", "decide_publish")
    # 从 decide_publish 分支到 publish 或 reject_publish
    # 关键：从 decide_publish 显式指向两个可能的下一步
    builder.add_edge("decide_publish", "publish")
    builder.add_edge("decide_publish", "reject_publish")

    builder.add_edge("publish", "post_publish_feedback")
    builder.add_edge("reject_publish", "done")
    builder.add_edge("post_publish_feedback", "done")

    # 持久化检查点
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph

def run_flow():
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "article_id": "art123",
        "article_text": "这是用户提交的一篇很长的文章内容……"
    }

    # 第一次 invoke，将会执行 generate_summary，然后暂停在 review_summary（中断）
    result = graph.invoke(initial_state, config=config)
    print("第一次中断：", result.get("__interrupt__"))

    # 假设人工修改摘要
    edited = "这是人工修改后的摘要版本。"
    result2 = graph.invoke(Command(resume=edited), config=config)
    # 这会执行 decide_publish，暂停等待人工“发布还是拒绝”
    print("第二次中断（发布决定）：", result2.get("__interrupt__"))

    # 假设人工批准发布
    result3 = graph.invoke(Command(resume="approve"), config=config)
    # 接着 publish node 会被调用，然后 post_publish_feedback node 中断等待反馈
    print("第三次中断（发布后反馈）：", result3.get("__interrupt__"))

    # 假设人工反馈“不满意，需要修改摘要再发布”
    feedback = "不太满意，摘要内容偏长希望简短点。"
    result4 = graph.invoke(Command(resume=feedback), config=config)
    # 这个例子中我们可以在 done node 打印状态或做进一步处理
    print("最终状态：", result4)

if __name__ == "__main__":
    run_flow()
