# 概念指引

>  TODO 注意目前仅完成了流式处理Streaming小节、多agent系统小节（包含agent之间通信）、。 

[参考文档：Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/#conceptual-guide)
本导航提供了更广泛的LangGraph框架和AI应用背后的核心概念说明。

我们推荐你在探索概念导航前至少经历过[快速开始](QuickStart.md)。这将提供更加实际的内容，让你能更早的理解这里讨论的概念。

本概念导航不会覆盖一步一步的介绍或明确的实现案例——那些是在 [教程](Tutorials.md) 和 [怎样做导航](HowtoGuides.md)。更多详细的参考资料，请看[API 参考](https://langchain-ai.github.io/langgraph/reference/)。

## Concepts[¶](https://langchain-ai.github.io/langgraph/concepts/#concepts)

- [为什么选择LangGraph？](https://langchain-ai.github.io/langgraph/concepts/high_level/): LangGraph高级概述和目标。
- [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/): LangGraph workflows are designed as graphs, with nodes representing different components and edges representing the flow of information between them. This guide provides an overview of the key concepts associated with LangGraph graph primitives.
- [Common Agentic Patterns](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/): An agent are LLMs that can pick its own control flow to solve more complex problems! Agents are a key building block in many LLM applications. This guide explains the different types of agent architectures and how they can be used to control the flow of an application.
- [多Agent系统](#multi-agent-system):复杂的大模型应用经常可用被分解成多agent，每个应用返回的内容都不一样。这个指引解释了构建多agent系统的常见模式。
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): Explains different ways of integrating human feedback into a LangGraph application.
- [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/): LangGraph has a built-in persistence layer, implemented through checkpointers. This persistence layer helps to support powerful capabilities like human-in-the-loop, memory, time travel, and fault-tolerance.
- [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/): Memory in AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and adapt to users' preferences.
- [Streaming](#Streaming): 流式输出是为增强用大模型构建应用程序的响应至关重要的。通过渐进的显示输出，甚至是在完整的响应之前就准备了，流式输出增强了用户的体验，尤其是当处理大模型的延迟时。
- [FAQ](https://langchain-ai.github.io/langgraph/concepts/faq/): Frequently asked questions about LangGraph.



<a id='multi-agent-system'></a>

## 多agen系统
[参考文档](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#multi-agent-systems)

Agent是一个用大模型来决定应用程序的控制流程的系统。跟你开发的那些系统一样，他们可能会随着时间的推移变得复杂，让他们变得难以管理和扩展。例如，你可能会出现如下问题：

- agent有太多的工具供其使用和工具调用下一步时做出不合理的决定
- 下文变更更复杂，单个agent难以追踪
- 系统中需要多个专业领域（例如：策划员、研究员、数学家等等）

为了应对这些，你可以考虑拆分你的应用变成多个小的，独立的并把它们构成**多agent系统**。这些独立的agent能够变得和一个大模型一个提示词一样简单，也可以变得跟 [ReAct](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation) agent一样复杂（等等）。

使用多agent系统的主要的好处有：

- **模块化**: 单独的agent更容易开发、测试和容易维护。
- **更专业**: 你能创建专业的agent，聚焦在专业领域，这能帮助整理的系统性能。
- **易控制**: .你可以明确的控制agent如何交互（而不是依赖函数调用）。

### 多agent系统架构 

![多agent系统结构图](./images/architectures.png)

这里有几种在多agent系统中的交互方式。

- **网状架构**: 每一个agent能够和其他所有的agent通信交流。任何agent都能决定之后去调用其他的agent。
- **监管架构**:  每一个agent只能和一个[监管](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)agent交互，监管agent做出确定哪一个agent应该被调用。
- **监管架构 (带工具调用)**：这是一个特殊的监管结构，单个agent可以表示为工具。在这个案例中，监管agent用工具调用大模型来决定哪一个agent的工具去调用，也将这些参数传递给那些agent。
- **分层管理架构**: 你可以定义一个包含[多层监管者](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)的多agent系统。这是监管架构的一个总体概况，允许你控制更为复杂的流程。
- **自定义多agent工作流**：每一个agent只能跟子agent交互。部分流程是确定性的，只有一些代理可以决定接下来调用哪个其他代理

### 网状架构

在这个结构中，agent被定义成graph的节点。每一个agent能够和其他的agent通信（多对多的连接），并且能够确定下一次应该调用哪个agent。这非常灵活，这种结构不能随着agent的增长而扩展：

- 很难决定接下来哪个agent应该被调用
- 很难确定多少[信息](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#shared-message-list)应该在agent系统直接被传递

我们建议在生产中避免使用这种体系结构，而是使用以下体系结构之一。

### 监管者架构

<a id="supervisor"></a>

在这种架构中，我们定义了agent作为节点并且添加了一个监管节点（大模型），这样就能决定下一步应该被调用的agent节点。我们用[条件边](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)根据监管者的决定将执行路由到合适的agent节点。这种架构也能让它自己很好的并行运行多agent或者[map-reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) 模式。

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

class AgentState(MessagesState):
    next: Literal["agent_1", "agent_2", "__end__"]

def supervisor(state: AgentState):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # the "next" key will be used by the conditional edges to route execution
    # to the appropriate agent
    return {"next": response["next_agent"]}

def agent_1(state: AgentState):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    return {"messages": [response]}

def agent_2(state: AgentState):
    response = model.invoke(...)
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node(supervisor)
builder.add_node(agent_1)
builder.add_node(agent_2)

builder.add_edge(START, "supervisor")
# route to one of the agents or exit based on the supervisor's decisiion
# if the supervisor returns "__end__", the graph will finish execution
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("agent_1", "supervisor")
builder.add_edge("agent_2", "supervisor")

supervisor = builder.compile()
```

查看这个[教程](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)，了解更多关于监管者多agent架构案例。

### 监管者架构 (工具调用)

在这个 [监管者](#supervisor) 的变种架构中，我们定义了独立的agent作为**工具**并且在监管者节点中使用带有工具调用的大模型。这样能实现作为带有两个节点的[ReaAct](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation)的agent— 一个是大模型节点（监管者），另外一个是能执行工具的工具调用节点 （在本案例中的agent）。

```python
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent

model = ChatOpenAI()

# this is the agent function that will be called as tool
# notice that you can pass the state to the tool via InjectedState annotation
def agent_1(state: Annotated[dict, InjectedState]):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    # return the LLM response as a string (expected tool response format)
    # this will be automatically turned to ToolMessage
    # by the prebuilt create_react_agent (supervisor)
    return response.content

def agent_2(state: Annotated[dict, InjectedState]):
    response = model.invoke(...)
    return response.content

tools = [agent_1, agent_2]
# the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
# that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
supervisor = create_react_agent(model, tools)
```

### 分层管理架构

当你添加更多agent到你的系统中时，使用监管者架构管理它们可能变得困难。监管架构可能开始调用下一个agent做出糟糕的决定，对单个监管者架构来记录他们来说，这个上下文可能变得太复杂。换言之，你最终遇到了在最初激发多agent系统同样的问题。
为了解决这个问题，你可以设计你的系统*管理层次*，例如，你可以创建一个独立的，专门的agent管理小组，通过单独的管理者结合最高决策者来管理这个小组。

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

# define team 1 (same as the single supervisor example above)
class Team1State(MessagesState):
    next: Literal["team_1_agent_1", "team_1_agent_2", "__end__"]

def team_1_supervisor(state: Team1State):
    response = model.invoke(...)
    return {"next": response["next_agent"]}

def team_1_agent_1(state: Team1State):
    response = model.invoke(...)
    return {"messages": [response]}

def team_1_agent_2(state: Team1State):
    response = model.invoke(...)
    return {"messages": [response]}

team_1_builder = StateGraph(Team1State)
team_1_builder.add_node(team_1_supervisor)
team_1_builder.add_node(team_1_agent_1)
team_1_builder.add_node(team_1_agent_2)
team_1_builder.add_edge(START, "team_1_supervisor")
# route to one of the agents or exit based on the supervisor's decisiion
# if the supervisor returns "__end__", the graph will finish execution
team_1_builder.add_conditional_edges("team_1_supervisor", lambda state: state["next"])
team_1_builder.add_edge("team_1_agent_1", "team_1_supervisor")
team_1_builder.add_edge("team_1_agent_2", "team_1_supervisor")

team_1_graph = team_1_builder.compile()

# define team 2 (same as the single supervisor example above)
class Team2State(MessagesState):
    next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]

def team_2_supervisor(state: Team2State):
    ...

def team_2_agent_1(state: Team2State):
    ...

def team_2_agent_2(state: Team2State):
    ...

team_2_builder = StateGraph(Team2State)
...
team_2_graph = team_2_builder.compile()


# define top-level supervisor

class TopLevelState(MessagesState):
    next: Literal["team_1", "team_2", "__end__"]

builder = StateGraph(TopLevelState)
def top_level_supervisor(state: TopLevelState):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which team to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_team" field)
    response = model.invoke(...)
    # the "next" key will be used by the conditional edges to route execution
    # to the appropriate team
    return {"next": response["next_team"]}

builder = StateGraph(TopLevelState)
builder.add_node(top_level_supervisor)
builder.add_node(team_1_graph)
builder.add_node(team_2_graph)

builder.add_edge(START, "top_level_supervisor")
# route to one of the teams or exit based on the supervisor's decision
# if the top-level supervisor returns "__end__", the graph will finish execution
builder.add_conditional_edges("top_level_supervisor", lambda state: state["next"])
builder.add_edge("team_1_graph", "top_level_supervisor")
builder.add_edge("team_2_graph", "top_level_supervisor")

graph = builder.compile()
```

### 自定义多agent流程

在这个架构中我们添加了独立的agent作为图的节点，并且提前定义了agent被调用的顺序。在LangGraph中有2中方式定义流程：

- **明确(静态)的控制流程（普通边）**：LangGraph允许你在应用中明确定义控制流程（例如：agent如何通信的顺序），通过[普通边](https://langchain-ai.github.io/langgraph/concepts/low_level/#normal-edges)实现。这是上面这种架构最确定的变种——我们总是提前知道哪个agent在下次被调用。
- **动态控制流程 (条件边)**: 在LangGraph中你可以允许大模型来决定你的应用的控制流程的一部分。这能通过[条件边](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)实现。有一个监管架构（带工具）的特殊的例子，在这个例子中，带工具调用的大模型增强了监管agent，将会决定工具（agent）调用顺序的。

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

def agent_1(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

def agent_2(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
# define the flow explicitly
builder.add_edge(START, "agent_1")
builder.add_edge("agent_1", "agent_2")
```

## agent之间的通信
[参考文档：](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#communication-between-agents)

最重要的事情是当创建一个多agent系统时弄清楚agent如何通信。有几种不同的考虑：

- agent是否[**通过graph状态或者通过工具调用**](#state-vs-calls)通信？
- 如何两个代理有[**不同的状态模式**](#different-status)怎么办？
- 如何通过[**共享message列表**](#shared-message-list)通信

### Graph状态对比工具调用
<a id="state-vs-calls"></a>

 [参考文档:Graph state vs tool calls](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#graph-state-vs-tool-calls)

在agent之间传输的“payload”是什么？在上面讨论的大多数架构中，agent通信是通过graph状态.在上面[带工具的监管架构]()案例中，"payload"是工具调用参数。

![区别](./images/request.png)

#### Graph 状态
[参考文档：Graph state](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#graph-state)

为了graph的状态通信，单个的agent需要被定义成graph节点。这些可以被添加为一个函数或者作为[子graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs)的入口。在graph执行的每一个步骤中，agent节点接收当前的graph状态，执行agent代码并且传递修改过的状态到下一个节点。

通常agent节点共享一个[状态模式](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)。然而，你可能想设计一个带有[不同状态模式](#different-status)的agent节点

### 不同的状态模式
<a id='different-status'></a>
[参考文档：Different state schemas](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#different-state-schemas)

一个agent可能需要与其他的agent有不同的状态模式。例如，一个搜索agent仅仅只需要记录查询和检索文档。在LangGraph中有两种方式实现：

- 定义一个带有单独状态模式的子graph agent。如果在子graph和父graph之间有不共享的key（渠道），这对添加[输入/输出转换](https://langchain-ai.github.io/langgraph/how-tos/subgraph-transform-state/)很重要，因此父graph需要知道如何与子graph通信。
- 定义带有[私有输入状态模式](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/)的agent节点函数。这样就允许传输仅仅需要执行特定agent的信息。

### 共享message列表
<a id='shared-message-list'></a>
[参考文档：Shared message list](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#shared-message-list)
为了agent通信，最通用的方式是通过一个共享的状态渠道，普通的message列表。假设在共享agent状态中总是至少有一个单独的渠道（key）。当通信时通过共享message列表时，有一个额外的考虑：agent应该共享[完整的历史思考过程](#share-full-history)还是仅仅只是[最终结果](#share-final-result)?

![区别](./images/response.png)

#### 共享完整历史

<a id="share-full-history"></a>

[参考文档：Share full history](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#share-full-history)

Agent能和其他agent共享它们思考过程的**完整历史**（例如:"scratchpad"）。 这个“scratchpad”通常看起来像一个[message列表](https://langchain-ai.github.io/langgraph/concepts/low_level/#why-use-messages)。分享完整的思考过程的好处是可能帮助其他agent做出更好的决定，和整体的提高系统的推理能力。坏处是随着agent的数量和它们的复杂度的增加，"scratchpad" 也会快速增加并且可能需要添加额外的[内存管理](https://langchain-ai.github.io/langgraph/concepts/memory/#managing-long-conversation-history)策略。

#### 共享最终结果

<a id="share-final-result"></a>

[参考文档：Share final result](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#share-final-result)

Agents 能够有它自己私有的“scratchpad”并且仅与其他的agent**共享最终结果**。这种方案可能对许多agent系统或者更复杂的agent更有效果。在本案例中，你可能需要定义[不同状态模式]()的agent。

对agent调用作为工具来说，管理者基于工具模式确定输入。另外，LangGraph允许在允许在运行时[传递状态](https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#pass-graph-state-to-tools)给独立的工具。所以下一级的agent能获取其父级的状态，如果需要的话。



<a id='Streaming'></a>

## Streaming

[参考文档：Streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/#streaming)

Langgraph以一流的流式输出支持构建的。在graph运行的时候，有几个不同的方式来流式输出。

### 流式的graph输出 (`.stream` 和 `.astream`)
[参考文档：Streaming graph outputs](https://langchain-ai.github.io/langgraph/concepts/streaming/#streaming-graph-outputs-stream-and-astream)

`.stream` 是 `.astream` 是同步和异步的方式来实现graph运行的流式输出。当调用方法（例如： `graph.stream(..., mode="...")`）时你有几种不同的模式可以指定。

- [`"values"`](https://langchain-ai.github.io/langgraph/how-tos/stream-values/): 这将在graph每一步执执行之后输出完整的状态值。
- [`"updates"`](https://langchain-ai.github.io/langgraph/how-tos/stream-updates/): 这将在graph每一步执执行之后输出更新的状态值。如果在同一个步骤中进行了多次更新，（例如多节点运行），则这些更新内容将分别进行流式输出。
- [`"custom"`](https://langchain-ai.github.io/langgraph/how-tos/streaming-content/): 这将流式输出你的graph节点内部自定义数据。
- [`"messages"`](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/): 当大模型被调用的时候，这将流式输出大模型的Token和元数据。
- `"debug"`:流式输出在整个graph执行期间尽可能多的信息。

你也可以在同样的时间点通过列表的形式指定多种流式模式。当你这样做之后，流式输出的数据将是一个元组(tuples,`(stream_mode, data)`)，例如：

```python
graph.stream(..., stream_mode=["updates", "messages"])
```

```
...
('messages', (AIMessageChunk(content='Hi'), {'langgraph_step': 3, 'langgraph_node': 'agent', ...}))
...
('updates', {'agent': {'messages': [AIMessage(content="Hi, how can I help you?")]}})
```

下面可视化展示了在`values`和`updates`模式下区别：

![区别](./images/values_vs_updates.png)

### 流式输出大模型的token和事件(`.astream_events`)

此外，你可以使用[`astream_events`](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/)方法来流式输出在内部节点的事件。这对流式输出大模型的token是非常有用的。

这是[LangChain 对象](https://python.langchain.com/docs/concepts/#runnable-interface)中的一个标准方法。这表示它作为graph被调用，如果你用`.astream_events`运行graph的时候，某些事件同时被发出并且能被看见。

所有的事件都有（但不限于）`event`, `name`, data` 字段，这些都是什么呢？

- `event`: 这是正在被发出的事件的一个类型，你可以在[这里](https://python.langchain.com/docs/concepts/#callback-events)找到所有回调事件的详细表格和触发器。
- `name`: 事件的名称。
- `data`: 事件所属的数据

什么类型的情况导致事件被发出？

- 当节点开始执行的时候，每一个节点（runnable)都发出`on_chain_start`，在节点执行期间发出 `on_chain_stream` 并且在节点完成的时候发出`on_chain_end` 。在事件`name`字段里面，节点事件将有节点名称。
- 在graph开始执行的手将发出`on_chain_start` ，之后每一个节点执行都发出`on_chain_stream` ，并且当graph完成的时候发出`on_chain_end` ，在事件`name`字段里面，graph事件将有`LangGraph`。
- 任何写入state状态的情况下（例如：每当你更新你的state的key的值时）都将发出`on_chain_start` 和`on_chain_end` 事件。

另外，任何你在节点内部创建的事件也将被显示在`.astream_events`的输出上。

为了更加具和看清楚里面是什么，让我们看看当运行一个简单的graph时返回什么样的事件：

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI(model="gpt-4o-mini")


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node(call_model)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)
app = workflow.compile()

inputs = [{"role": "user", "content": "hi!"}]
async for event in app.astream_events({"messages": inputs}, version="v1"):
    kind = event["event"]
    print(f"{kind}: {event['name']}")
```

```
on_chain_start: LangGraph
on_chain_start: __start__
on_chain_end: __start__
on_chain_start: call_model
on_chat_model_start: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_end: ChatOpenAI
on_chain_start: ChannelWrite<call_model,messages>
on_chain_end: ChannelWrite<call_model,messages>
on_chain_stream: call_model
on_chain_end: call_model
on_chain_stream: LangGraph
on_chain_end: LangGraph
```

我们从总体的graph开始（`on_chain_start: LangGraph`）。之后写入`__start__` 节点（这是一个特殊的节点来处理输入）。我们之后从`call_model` 节点开始（`on_chain_start: call_model`）.然后从聊天模式调用开始（`on_chat_model_start: ChatOpenAI`），流式返回token（`on_chat_model_stream: ChatOpenAI`）并且完成聊天模式（`on_chat_model_end: ChatOpenAI`）。到这里，我们写回结果到通道（`ChannelWrite<call_model,messages>`）并且完成了`call_model` 节点，然后完成整个graph。

在一个简单的graph里面，这应该有希望给你一个好的判断，看看什么事件被发出。但是这些事件里面包含了什么样的数据呢？每一种格式的事件包含的数据格式不一样。让我们看看`on_chat_model_stream` 事件里面的数据。这是一个很重要的事件类型，因为它是从大模型响应流式输出token时所必须的。

这些事件像这样：

```python
{'event': 'on_chat_model_stream',
 'name': 'ChatOpenAI',
 'run_id': '3fdbf494-acce-402e-9b50-4eab46403859',
 'tags': ['seq:step:1'],
 'metadata': {'langgraph_step': 1,
  'langgraph_node': 'call_model',
  'langgraph_triggers': ['start:call_model'],
  'langgraph_task_idx': 0,
  'checkpoint_id': '1ef657a0-0f9d-61b8-bffe-0c39e4f9ad6c',
  'checkpoint_ns': 'call_model',
  'ls_provider': 'openai',
  'ls_model_name': 'gpt-4o-mini',
  'ls_model_type': 'chat',
  'ls_temperature': 0.7},
 'data': {'chunk': AIMessageChunk(content='Hello', id='run-3fdbf494-acce-402e-9b50-4eab46403859')},
 'parent_ids': []}
```

我们可以看到我们有事件的类型和名称（这些在之前我们就知道）

我们在元素据里面也有一堆东西。显然，`'langgraph_node': 'call_model',`是一些真正有用的信息，能告诉我们模型在哪个节点内部被调用。

最后，`data` 是一个非常重要的字段。这包含了真实的来自事件数据！在这个例子里面是`AIMessageChunk`。这包含了消息的`content` ，也有`id`。这是总体`AIMessage `（不仅仅是这个单独的块）的ID，真的是超级有用——这能帮助我们追踪块属于相同消息的一部分（所以我们能够一起展示到UI上面）。

这个信息包含了所有创建流式输出token的UI界面必须的信息。你可以看[这里](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)的指引。

**ASYNC IN PYTHON<=3.10**

> You may fail to see events being emitted from inside a node when using `.astream_events` in Python <= 3.10. If you're using a Langchain RunnableLambda, a RunnableGenerator, or Tool asynchronously inside your node, you will have to propagate callbacks to these objects manually. This is because LangChain cannot automatically propagate callbacks to child objects in this case. Please see examples [here](https://langchain-ai.github.io/langgraph/how-tos/streaming-content/) and [here](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/).

### LangGraph 框架

[参考文档：LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/streaming/#langgraph-platform)

流式输出是让大模型应用感受响应给终端用户非常重要的。当创建一个流式运行时，流式模式说明数据是流式返回给客户端API的。LangGraph框架支持5种流式模式。

- `values`: Stream the full state of the graph after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) is executed. See the [how-to guide](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_values/) for streaming values.
- `messages`: 流式输出完整的消息（在节点执行最后）以及节点内部生成的任何消息token。 这个模式主要表示增强聊天程序。这只是一个选项，如果你的的grap里面包含了一个`messages` 。见 [how-to-guide][./HowtoGuides.md]流式输出message。
- `updates`: 当每一个节点被执行的时候，流式输出graph内状态更新的内容，见[how-to-guide][./HowtoGuides.md]流式输出的updates.
- `events`: 流式输出所有在graph执行期间的事件（包含graph的状态）。见[how-to-guide][./HowtoGuides.md]流式输出的事件。这种模式可用被用作大模型一个token一个token返回的流式输出。 
- `debug`: 流式调试grap执行的所有事件。见 [how-to guide](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_debug/)流式输出debug事件。

你也可以指定同时指定多种流式模式。见[how-to-guide][./HowtoGuides.md]同时配置多种流式模式。

见 [API 手册](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html#tag/threads-runs/POST/threads/{thread_id}/runs/stream) 怎样创建运行流式模式。

流式模式种的values、updates、messages-tuple和debug在LangGraph库中可用的模式与是非常相似的——为了更加深入的说明这些，你可用见上一章节。

流式模式的events是和LangGraph库种的`astream_events`相似的——为了更加深入的说明这些，你可用见上一章节。

所有的事件发出有两个属性：

event: 这是事件的名称
data: 这是关联事件的数据