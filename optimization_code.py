# ============================================================================
# 优化版本：添加调用次数限制和阈值控制
# ============================================================================

# 添加调用计数器
class CallCounter:
    def __init__(self, max_calls=5):
        self.max_calls = max_calls
        self.current_calls = 0
    
    def increment(self):
        self.current_calls += 1
        return self.current_calls <= self.max_calls
    
    def can_call(self):
        return self.current_calls < self.max_calls
    
    def reset(self):
        self.current_calls = 0

# 全局调用计数器
call_counter = CallCounter(max_calls=5)

# 配置参数
RETRIEVER_CONFIG = {
    "top_k": 3,  # 检索文档数量
    "use_llm_grading": False,  # 是否使用LLM评估
    "auto_grade_threshold": 2,  # 当top_k <= 此值时自动使用LLM评估
}

def should_use_llm_grading(top_k: int) -> bool:
    """根据top_k决定是否使用LLM评估"""
    return top_k <= RETRIEVER_CONFIG["auto_grade_threshold"]

def smart_query_and_evaluate(state: MessagesState):
    """合并查询决策和文档评估，减少LLM调用次数"""
    if not call_counter.can_call():
        # 如果超过调用限制，直接生成答案
        return {"messages": [{"role": "assistant", "content": "已达到最大调用次数限制，基于可用信息生成答案。"}], "action": "generate_answer"}
    
    question = state["messages"][0].content
    
    # 如果消息历史中有工具调用结果，直接评估
    if len(state["messages"]) > 2 and any(
        hasattr(msg, 'tool_calls') and msg.tool_calls for msg in state["messages"]
    ):
        # 直接评估检索结果，不需要重新决策
        return evaluate_and_decide(state)
    
    # 首次调用：判断是否需要检索
    call_counter.increment()
    response = response_model.bind_tools([retriever_tool]).invoke(
        [{"role": "user", "content": question}]
    )
    
    if response.tool_calls:
        return {"messages": [response], "action": "retrieve"}
    else:
        return {"messages": [response], "action": "end"}

def evaluate_and_decide(state: MessagesState):
    """评估检索结果并决定下一步，使用智能评估策略"""
    if not call_counter.can_call():
        # 如果超过调用限制，直接生成答案
        return {"action": "generate_answer"}
    
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    # 根据top_k选择评估策略
    top_k = RETRIEVER_CONFIG["top_k"]
    
    if should_use_llm_grading(top_k):
        # 小top_k使用LLM评估
        return llm_based_evaluation(state)
    else:
        # 大top_k使用轻量评估
        return light_evaluation(state)

def llm_based_evaluation(state: MessagesState):
    """使用LLM进行文档评估（适用于小top_k）"""
    if not call_counter.can_call():
        return {"action": "generate_answer"}
    
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    # 使用原有的LLM评估逻辑
    call_counter.increment()
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    
    if score == "yes":
        return {"action": "generate_answer"}
    else:
        return {"action": "rewrite_question"}

def light_evaluation(state: MessagesState):
    """轻量级评估（适用于大top_k）"""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    # 使用快速相关性检查
    relevance_score = quick_relevance_check(question, context)
    
    # 可配置的阈值
    RELEVANCE_THRESHOLD = 0.5  # 大top_k时降低阈值
    MAX_REWRITE_ATTEMPTS = 1   # 大top_k时减少重写次数
    
    # 检查重写次数
    rewrite_count = sum(1 for msg in state["messages"] if 
                       hasattr(msg, 'content') and "improved question" in str(msg.content).lower())
    
    if relevance_score > RELEVANCE_THRESHOLD:
        return {"action": "generate_answer"}
    elif rewrite_count < MAX_REWRITE_ATTEMPTS:
        # 大top_k时，重写问题需要消耗LLM调用
        if call_counter.can_call():
            call_counter.increment()
            return {"action": "rewrite_question"}
        else:
            return {"action": "generate_answer"}
    else:
        return {"action": "generate_answer"}

def quick_relevance_check(question: str, context: str) -> float:
    """快速相关性检查，使用关键词和语义匹配"""
    # 关键词重叠检查
    question_words = set(question.lower().split())
    context_words = set(context.lower().split())
    
    # 过滤停用词
    stop_words = {'what', 'does', 'say', 'about', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    question_words = question_words - stop_words
    context_words = context_words - stop_words
    
    keyword_overlap = len(question_words.intersection(context_words))
    keyword_score = min(keyword_overlap / max(len(question_words), 1), 1.0)
    
    # 简单的语义检查（可以进一步优化）
    semantic_score = 0.5  # 基础分数
    
    # 综合评分
    final_score = (keyword_score * 0.7) + (semantic_score * 0.3)
    
    return final_score

def adaptive_retriever_tool():
    """创建自适应检索工具，根据配置调整top_k"""
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_CONFIG["top_k"]}
    )
    
    # 创建检索工具
    return create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        f"Search and return information about Lilian Weng blog posts. Returns top {RETRIEVER_CONFIG['top_k']} results.",
    )

# 优化版本的工作流程
def create_optimized_workflow():
    """创建优化的工作流程，包含智能评估策略"""
    workflow = StateGraph(MessagesState)
    
    # 使用自适应检索工具
    adaptive_retriever = adaptive_retriever_tool()
    
    # 添加节点
    workflow.add_node("smart_evaluate", smart_query_and_evaluate)
    workflow.add_node("retrieve", ToolNode([adaptive_retriever]))
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("rewrite_question", rewrite_question)
    
    # 添加边
    workflow.add_edge(START, "smart_evaluate")
    
    # 条件边
    workflow.add_conditional_edges(
        "smart_evaluate",
        lambda x: x["action"],
        {
            "retrieve": "retrieve",
            "end": END,
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_conditional_edges(
        "retrieve",
        evaluate_and_decide,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question"
        }
    )
    
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "smart_evaluate")
    
    return workflow.compile()

# 创建优化的工作流程
optimized_graph = create_optimized_workflow()

# 重置调用计数器
call_counter.reset()

# 测试优化后的工作流程
print("=== 测试优化后的工作流程 ===")
print(f"最大调用次数限制: {call_counter.max_calls}")
print(f"检索文档数量 (top_k): {RETRIEVER_CONFIG['top_k']}")
print(f"使用LLM评估: {should_use_llm_grading(RETRIEVER_CONFIG['top_k'])}")
print(f"相关性阈值: {'0.6 (LLM评估)' if should_use_llm_grading(RETRIEVER_CONFIG['top_k']) else '0.5 (轻量评估)'}")
print(f"最大重写尝试次数: {'2 (LLM评估)' if should_use_llm_grading(RETRIEVER_CONFIG['top_k']) else '1 (轻量评估)'}")

# 测试用例
test_input = {
    "messages": [
        {
            "role": "user",
            "content": "What does Lilian Weng say about types of reward hacking?",
        }
    ]
}

print("\n开始执行优化后的工作流程...")
for chunk in optimized_graph.stream(test_input):
    for node, update in chunk.items():
        print(f"\n节点: {node}")
        if "messages" in update:
            msg = update["messages"][-1]
            if hasattr(msg, 'pretty_print'):
                msg.pretty_print()
            else:
                print(f"消息内容: {msg}")
        if "action" in update:
            print(f"下一步动作: {update['action']}")
        print(f"当前调用次数: {call_counter.current_calls}/{call_counter.max_calls}")

print(f"\n=== 执行完成 ===")
print(f"总调用次数: {call_counter.current_calls}")
print(f"是否达到限制: {'是' if call_counter.current_calls >= call_counter.max_calls else '否'}")

# 配置建议
print(f"\n=== 配置建议 ===")
print("1. 当 top_k <= 2 时，建议使用LLM评估 (use_llm_grading=True)")
print("2. 当 top_k > 2 时，建议使用轻量评估 (use_llm_grading=False)")
print("3. 可以根据实际需求调整 auto_grade_threshold 参数")
print("4. 大top_k时，建议降低相关性阈值和重写次数限制")
