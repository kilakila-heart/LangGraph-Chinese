# 概念导航

>  TODO 注意目前仅完成了流式处理Streaming这个小节。 

[参考文档：Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/#conceptual-guide)
本导航提供了更广泛的LangGraph框架和AI应用背后的核心概念说明。

我们推荐你在探索概念导航前至少经历过[快速开始](QuickStart.md)。这将提供更加实际的内容，让你能更早的理解这里讨论的概念。

本概念导航不会覆盖一步一步的介绍或明确的实现案例——那些是在 [教程](Tutorials.md) 和 [怎样做导航](HowtoGuides.md)。更多详细的参考资料，请看[API 参考](https://langchain-ai.github.io/langgraph/reference/)。

## Concepts[¶](https://langchain-ai.github.io/langgraph/concepts/#concepts)

- [为什么选择LangGraph？](https://langchain-ai.github.io/langgraph/concepts/high_level/): LangGraph高级概述和目标。
- [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/): LangGraph workflows are designed as graphs, with nodes representing different components and edges representing the flow of information between them. This guide provides an overview of the key concepts associated with LangGraph graph primitives.
- [Common Agentic Patterns](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/): An agent are LLMs that can pick its own control flow to solve more complex problems! Agents are a key building block in many LLM applications. This guide explains the different types of agent architectures and how they can be used to control the flow of an application.
- [Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/): Complex LLM applications can often be broken down into multiple agents, each responsible for a different part of the application. This guide explains common patterns for building multi-agent systems.
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): Explains different ways of integrating human feedback into a LangGraph application.
- [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/): LangGraph has a built-in persistence layer, implemented through checkpointers. This persistence layer helps to support powerful capabilities like human-in-the-loop, memory, time travel, and fault-tolerance.
- [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/): Memory in AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and adapt to users' preferences.
- [Streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/): 流式输出是为增强用大模型构建应用程序的响应至关重要的。通过渐进的显示输出，甚至是在完整的响应之前就准备了，流式输出增强了用户的体验，尤其是当处理大模型的延迟时。
- [FAQ](https://langchain-ai.github.io/langgraph/concepts/faq/): Frequently asked questions about LangGraph.



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