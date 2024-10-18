# langGraph的中文指引
源文档地址：[原文档地址](https://langchain-ai.github.io/langgraph/)

langchain agent如何迁移到LangGraph：https://python.langchain.com/v0.2/docs/how_to/migrate_agent/



>由于使用传统的langchain的AgentExecutor 构建agent没有的灵活性和控制力，langchain官方已经推荐使用langGraph来创建根据灵活易用的langGraph来创建agent，并编写了从langchian的agent迁移到langGraph的教程，可见日后使用langGraph构建agent将会作为langchain团队的重心工作之一。
>因此本项目将特地翻译LangGraph的文档。


## 概述

LangGraph(https://langchain-ai.github.io/langgraph/)  是一个python库，用于构建有状态的，多操作的大模型（LLM）应用程序，用于创建agent（智能体）和multi-agent（组合智能体）流程。和其他的LLM应用框架相比，他提供了核心的优点：循环、可控的和持久化。LangGraph 允许你自定义涉及到循环的流程，这对大多数agent架构来说都是必不可少的，这使它有别于基于DAG的解决方案。作为一个底层的框架，LangGraph为你提供了涉及到流程和状态的应用程序更细颗粒的控制，这对创建可靠的agent应用来说至关重要。另外，LangGraph 包括内置的持久化功能，能支持高级的人工介入(在智能体执行过程中)和记忆功能。

LangGraph 的灵感来源于 [Pregel](https://research.google/pubs/pub37252/) 和 [Apache Beam](https://beam.apache.org/).公共接口灵感来自于 [NetworkX](https://networkx.org/documentation/latest/). LangGraph是基于LangChain的，但是也可以在没有LangChain的环境下使用。

学习更多LangGraph知识，请查阅我们的第一个LangChain 学院课程，*Introduction to LangGraph*, 免费使用 [here](https://academy.langchain.com/courses/intro-to-langgraph).



### [关键特性](https://langchain-ai.github.io/langgraph/#key-features)

- **循环和分支控制**: 在你的应用中可以实现循环和判断条件的控制.
- **持久化**: 在graph（图）的每一个步骤中自动保存。随时暂停和恢复graph的执行，以支持错误恢复、人工介入、时间旅行（从指定的点重新执行）等。
- **人工介入**: 通过agent，能够中断graph的执行来确认或者编辑下一个动作计划。
- **流失支持**: 每一个节点都能以流式输出 (包括token的流失输出).
- **与langchain进行整合**: LangGraph 可以和 [LangChain](https://github.com/langchain-ai/langchain/) 、 [LangSmith](https://docs.smith.langchain.com/)无缝集成 (但不是必须依赖).

## **初始化**

```bash
pip install -U langgraph
```
## **示例**
LangGraph 中一个核心概念是状态(state)。每一个graph执行都会创建一个state，图在传输节点之间执行时，每一个节点执行之后都会在内容部更新这个state并将其返回。graph在内部更新state的方式是由graph的选择或自定义函数（function）定义。

让我们看一个能用搜索工具的简单agent例子。

```bash
pip install langchain-anthropic
export ANTHROPIC_API_KEY=sk-...
```


同时，我们可以设置[LangSmith](https://docs.smith.langchain.com/) ，以实现最佳的观察体验。

```python
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_sk_...
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]

tool_node = ToolNode(tools)

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).bind_tools(tools)

# Define the function that determines whether to continue or not
# 定义一个函数确定是否继续执行
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    # 如果大模型通知调用工具的时候，我们可以路由到对应的工具节点
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user) 否则，停止执行（回复用户）
    return END


# Define the function that calls the model 定义一个调用大模型的函数
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph 定义一个图
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between 定义两个可以循环的节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent` 定义agent的入口
# This means that this node is the first one called 这表示这是第一个被调用的节点
workflow.add_edge(START, "agent")

# We now add a conditional edge 添加条件dege（边）
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called. 这表示这些边在`agent`节点调用之后执行
    "agent",
    # Next, we pass in the function that will determine which node is called next. 接下来通过这个函数决定哪一个节点将被调用
    should_continue,
)

# We now add a normal edge from `tools` to `agent`. 从工具到agent中添加一个普通的边（edge）
# This means that after `tools` is called, `agent` node is called next.
# 这表示tools工具被调用后，紧接着调用agent节点
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
# 初始化内从以保存graph之间的允许
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
# 最后编译，编译成一个langchain的runnable，意味着你可以像使用其他任意的runnable一样使用他，注意我们在刚刚编译的时候放入了内存记忆（memory)
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
"Based on the search results, I can tell you that the current weather in San Francisco is:\n\nTemperature: 60 degrees Fahrenheit\nConditions: Foggy\n\nSan Francisco is known for its microclimates and frequent fog, especially during the summer months. The temperature of 60°F (about 15.5°C) is quite typical for the city, which tends to have mild temperatures year-round. The fog, often referred to as "Karl the Fog" by locals, is a characteristic feature of San Francisco\'s weather, particularly in the mornings and evenings.\n\nIs there anything else you\'d like to know about the weather in San Francisco or any other location?"
```

现在我们放入同样的`"thread_id"`，上下文对话记录通过保存状态state记忆下来（即存储在消息列表中）
```
final_state = app.invoke(
    {"messages": [HumanMessage(content="what about ny")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
"Based on the search results, I can tell you that the current weather in New York City is:\n\nTemperature: 90 degrees Fahrenheit (approximately 32.2 degrees Celsius)\nConditions: Sunny\n\nThis weather is quite different from what we just saw in San Francisco. New York is experiencing much warmer temperatures right now. Here are a few points to note:\n\n1. The temperature of 90°F is quite hot, typical of summer weather in New York City.\n2. The sunny conditions suggest clear skies, which is great for outdoor activities but also means it might feel even hotter due to direct sunlight.\n3. This kind of weather in New York often comes with high humidity, which can make it feel even warmer than the actual temperature suggests.\n\nIt's interesting to see the stark contrast between San Francisco's mild, foggy weather and New York's hot, sunny conditions. This difference illustrates how varied weather can be across different parts of the United States, even on the same day.\n\nIs there anything else you'd like to know about the weather in New York or any other location?"
```

### [逐步分解以上步骤](https://langchain-ai.github.io/langgraph/#step-by-step-breakdown)

待完成TODO

# [文档](https://langchain-ai.github.io/langgraph/#documentation)

- [教程](Tutorials.md):  通过学习指引的例子构建LangGraph。
- [操作指南](HowtoGuides.md): 
  利用LangGraph中的流、添加记忆和持久化功能，利用通用的设计模式（分支、子图等等）完成特定的事情，这是一个你可以去复制和运行指定的代码片段的地方。
- [概念导航](https://langchain-ai.github.io/langgraph/concepts/high_level/): 深入解释LangGraph背后的关键概念和原理，如节点(nodes)、边(edges)、状态(state )等。
- [API 文档](https://langchain-ai.github.io/langgraph/reference/graphs/): 
  仔细研究重要的类和方法，一些如何使用图和切入点API的简单示例，更高级的预构建组件等。
- [云服务 (beta)](https://langchain-ai.github.io/langgraph/cloud/): 一键将LangGraph应用部署到LangGraph Cloud上。

