# How-to guides(操作指南)
[官网](https://langchain-ai.github.io/langgraph/how-tos/#how-to-guides)

注意：TODO 本人目前重点关注了stream章节，其他的尚未完成，请先关注官网

欢迎来到LangGraph的操作指南，本指南提供实用的，一步一步的介绍来完成LangGraph中的关键任务。

## 可控性
LangGraph is known for being a highly controllable agent framework. These how-to guides show how to achieve that controllability.

- [How to create branches for parallel execution](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [How to create map-reduce branches for parallel execution](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [How to control graph recursion limit](https://langchain-ai.github.io/langgraph/how-tos/recursion-limit/)



## Persistence[¶](https://langchain-ai.github.io/langgraph/how-tos/#persistence)

LangGraph makes it easy to persist state across graph runs (thread-level persistence) and across threads (cross-thread persistence). These how-to guides show how to add persistence to your graph.

- [How to add thread-level persistence to your graph](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [How to add cross-thread persistence to your graph](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/)
- [How to use Postgres checkpointer for persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)
- [How to create a custom checkpointer using MongoDB](https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/)
- [How to create a custom checkpointer using Redis](https://langchain-ai.github.io/langgraph/how-tos/persistence_redis/)



## Memory[¶](https://langchain-ai.github.io/langgraph/how-tos/#memory)

LangGraph makes it easy to manage conversation [memory](https://langchain-ai.github.io/langgraph/concepts/memory/) in your graph. These how-to guides show how to implement different strategies for that.

- [How to manage conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
- [How to delete messages](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/)
- [How to add summary conversation memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/)



## Human in the Loop[¶](https://langchain-ai.github.io/langgraph/how-tos/#human-in-the-loop)

One of LangGraph's main benefits is that it makes human-in-the-loop workflows easy. These guides cover common examples of that.

- [How to add breakpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [How to add dynamic breakpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/dynamic_breakpoints/)
- [How to edit graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/)
- [How to wait for user input](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [How to view and update past graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/)
- [Review tool calls](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/)



## Streaming流式传输

 [参考文档](https://langchain-ai.github.io/langgraph/how-tos/#streaming)

LangGraph首先被构建为流式交互。本指南展示怎样使用不同的流式streaming模式。

- [怎样流式处理完整的graph状态state](#steam1)
- [怎样使用updates模式流式处理graph状态](#How-to-stream-state-updates-of-your-graph)
- [怎样在大模型流式输出tokens](#How-to-stream-LLM-tokens-from-your-graph)
- [怎样在不用Langchain大模型API下对大模型进行steam输出](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens-without-langchain/)
- [怎样steam传输自定义数据](https://langchain-ai.github.io/langgraph/how-tos/streaming-content/)
- [怎样同时配置多steam式传输](https://langchain-ai.github.io/langgraph/how-tos/stream-multiple/)
- [如何在工具中流式输出事件](#steam7)
- [How to stream events from within a tool without LangChain models](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools-without-langchain/)
- [How to stream events from the final node 怎样从最终节点传输(stream)事件](https://langchain-ai.github.io/langgraph/how-tos/streaming-from-final-node/)
- [How to stream from subgraphs](https://langchain-ai.github.io/langgraph/how-tos/streaming-subgraphs/)
- [How to disable streaming for models that don't support it](https://langchain-ai.github.io/langgraph/how-tos/disable-streaming/)



### 怎样流式处理完整的graph状态（values） 
<a id='steam1'></a>
[源文档：How to stream full state of your graph](https://langchain-ai.github.io/langgraph/how-tos/stream-values/#how-to-stream-full-state-of-your-graph)

LangGraph支持多种流模式。主要的有:

values:  这种流式模式返回graph的值。这是每一个节点被调用后**完整的graph状态（state）**。
updates：这种流式模式返回graph更新后的内容。这是每一个节点被调用后**更新后的graph状态（state）**

本指南包含了流式模式`stream_mode="values"`

#### 准备

首先，按照相关依赖包和设置你自己的API key。

```python
%%capture --no-stderr
%pip install -U langgraph langchain-openai langchain-community
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

>Set up [LangSmith](https://smith.langchain.com/) for LangGraph development

> Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com/).



####  定义graph
[源文档链接](https://langchain-ai.github.io/langgraph/how-tos/stream-values/#define-the-graph)

本示例我们将使用简单的ReAct模式的agent。
```python
from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
graph = create_react_agent(model, tools)
```
参考API: [TavilySearchResults](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.tavily_search.tool.TavilySearchResults.html) | [ConfigurableField](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.utils.ConfigurableField.html) | [tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) | [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) | [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)

####  Stream values模式

[源文档链接](https://langchain-ai.github.io/langgraph/how-tos/stream-values/#stream-values)



```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph.astream(inputs, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```
```text
================================[1m Human Message [0m=================================

what's the weather in sf
==================================[1m Ai Message [0m==================================
Tool Calls:
  get_weather (call_61VvIzqVGtyxcXi0z6knZkjZ)
 Call ID: call_61VvIzqVGtyxcXi0z6knZkjZ
  Args:
    city: sf
=================================[1m Tool Message [0m=================================
Name: get_weather

It's always sunny in sf
==================================[1m Ai Message [0m==================================

The weather in San Francisco is currently sunny.
```

如果我们只想获取最终结果，我们可以使用通用的方法并只需要追踪我们接收到的最后一个值
```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph.astream(inputs, stream_mode="values"):
    final_result = chunk
```
```python
final_result
```
```text
{'messages': [HumanMessage(content="what's the weather in sf", id='54b39b6f-054b-4306-980b-86905e48a6bc'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_avoKnK8reERzTUSxrN9cgFxY', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_5e6c71d4a8', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-f2f43c89-2c96-45f4-975c-2d0f22d0d2d1-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_avoKnK8reERzTUSxrN9cgFxY'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
  ToolMessage(content="It's always sunny in sf", name='get_weather', id='fc18a798-c7b2-4f73-84fa-8ffdffb6ddcb', tool_call_id='call_avoKnK8reERzTUSxrN9cgFxY'),
  AIMessage(content='The weather in San Francisco is currently sunny. Enjoy the sunshine!', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 84, 'total_tokens': 98}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_5e6c71d4a8', 'finish_reason': 'stop', 'logprobs': None}, id='run-21418147-da8e-4738-a076-239377397c40-0', usage_metadata={'input_tokens': 84, 'output_tokens': 14, 'total_tokens': 98})]}
```

```python
final_result["messages"][-1].pretty_print()
```

```tex
==================================[1m Ai Message [0m==================================

The weather in San Francisco is currently sunny. Enjoy the sunshine!
```



###  怎样流式处理graph的状态更新(updates) 
<a id="How-to-stream-state-updates-of-your-graph"></a>

 [源文档How to stream state updates of your graph](https://langchain-ai.github.io/langgraph/how-tos/stream-updates/#how-to-stream-state-updates-of-your-graph)

LangGraph支持多种流模式。主要的有:

values:  这种流式模式返回graph的值。这是每一个节点被调用后**完整的graph状态（state）**。
updates：这种流式模式返回graph更新后的内容。这是每一个节点被调用后**更新后的graph状态（state）**

本指南包含了流式模式`stream_mode="updates"`。
#### 准备

首先，按照相关依赖包和设置你自己的API key。

```python
%%capture --no-stderr
%pip install -U langgraph langchain-openai langchain-community
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

>Set up [LangSmith](https://smith.langchain.com/) for LangGraph development

> Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com/).

####  定义graph
[源文档链接](https://langchain-ai.github.io/langgraph/how-tos/stream-values/#define-the-graph)

本示例我们将使用简单的ReAct模式的agent。
```python
from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
graph = create_react_agent(model, tools)
```
参考API: [TavilySearchResults](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.tavily_search.tool.TavilySearchResults.html) | [ConfigurableField](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.utils.ConfigurableField.html) | [tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) | [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) | [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)

####  Stream updates模式
[源文档](https://langchain-ai.github.io/langgraph/how-tos/stream-updates/#stream-updates)

```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        print(f"Receiving update from node: '{node}'")
        print(values)
        print("\n\n")
```
```text
Receiving update from node: 'agent'
{'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_kc6cvcEkTAUGRlSHrP4PK9fn', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3e7d703517', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd68b3a0-86c3-4afa-9649-1b962a0dd062-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_kc6cvcEkTAUGRlSHrP4PK9fn'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})]}



Receiving update from node: 'tools'
{'messages': [ToolMessage(content="It's always sunny in sf", name='get_weather', tool_call_id='call_kc6cvcEkTAUGRlSHrP4PK9fn')]}



Receiving update from node: 'agent'
{'messages': [AIMessage(content='The weather in San Francisco is currently sunny.', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3e7d703517', 'finish_reason': 'stop', 'logprobs': None}, id='run-009d83c4-b874-4acc-9494-20aba43132b9-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}
```



###  怎样在你的graph中流式输出大模型(LLM)的tokens 
<a id="How-to-stream-LLM-tokens-from-your-graph"></a>
[参考源文档： How to stream LLM tokens from your graph](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#how-to-stream-llm-tokens-from-your-graph)

在本例子中我们将流式输出大语言模型的tokens来增强agent。我们将使用ReAct模式的agent作为例子。

本指南紧跟着本目录中的其他指南，所以我们将用下面的STREAMING标签指出不同之处（如果你只是想搜索他们的差异）

注意

>In this how-to, we will create our agent from scratch to be transparent (but verbose). You can accomplish similar functionality using the create_react_agent(model, tools=tool) (API doc) constructor. This may be more appropriate if you are used to LangChain’s AgentExecutor class.



关于 Python < 3.11的说明

>当你使用的python版本是3.8，3.9或者3.10时，请确保当调用大模型时需要手动传入RunnableConfig 给它，例如：llm.ainvoke(...,config).这个流式方法从你嵌套的代码中用流式追踪器传递给回调函数来搜集了所有的事件。在3.11及以上版本中，是通过contextvar自动处理的，在3.11之前，asyncio的任务缺乏适当的contextvar支持，这意味着回调只有在您手动传递配置时才会传播。我们在下面的call_model方法中实现了这一点。

#### 准备
我们首先安装依赖包

```bash
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langsmith
```
Next, we need to set API keys for OpenAI (the LLM we will use).

```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```
为LangGraph 开发模式设置LangSmith 

> Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started here.

#### 设置状态state
[参考源文档：Set up the state](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#set-up-the-state)


在 `langgraph` 的主要类是 [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph). 这个graph由`State`对象参数化，它将传输到每一个节点。然后每个节点返回操作，这样graph就用来`update`该状态state。这些操作能够既能设置指定状态state属性（例如覆盖已存在的值），也能追加到已存在的属性中。无论是重置还是追加都是通过你构造graph时候的`State`对象来标注的。

对这个例子，我们仅仅将记录这个状态来作为messages列表的一部分。我们希望每一个节点仅仅添加到messages的列表中。因此，我们将用一个`TypedDict`作为一个key（`messages`）来标注它，所以`messages`属性是”仅追加“。

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

# Add messages essentially does this with more
# robust handling
# def add_messages(left: list, right: list):
#     return left + right


class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**API 参考:** [add_messages](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)

####  设置工具tools
[参考文档：Set up the tools](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#set-up-the-tools)


首先定义我们想使用的工具。对这个简单例子，我们将创建一个模拟的搜索引擎。这能简单的创建你自己的工具——看[这里](https://python.langchain.com/docs/how_to/custom_tools)的文档是怎么做的。

```python
from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    return ["Cloudy with a chance of hail."]


tools = [search]
```

**API 参考:** [tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html)

我们现在能用简单的[ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)包装这些工具了。这是一个简单的类，在messages列表中包含了带有[tool_calls的AIMessages]((https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.tool_calls))，运行这个工具，并返回输出作为[ToolMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolMessage.html#langchain_core.messages.tool.ToolMessage)

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

**API 参考:** [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)



#### 设置大模型model
[参考文档：Set up the model](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#set-up-the-model)


现在我们需要加载我们想使用的大模型。这应该满足两个条件：

 1. 它应该能处理消息messages，因为我们的状态state主要由消息列表messages构成（聊天历史）
 2. 它应能处理工具调用，因为我们使用了预构建的[ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)

**注意:** 模型依赖不是必须使用LangGraph的。 —— 只是这个特殊例子的要求。

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
```

**API 参考:** [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)


我们做完这步后，应该确保大模型知道它有这些工具可以调用。我们可以用覆盖langchian工具转换成函数调用的格式实现这一点，然后将他们绑定到mode的类上，如下代码：

```python
model = model.bind_tools(tools)
```

#### 定义节点nodes
[参考：Define the nodes](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#define-the-nodes)



现在我们需要在graph中定义几个不同的节点，在`langgraph`中，一个节点可以是一个函数，也可以是一个[runnable](https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel). 这里我们需要两个主要的节点：

1. agent 节点：返回决定需要采取的行动（如果有）。
2. 一个调用工具的函数：如果agent需要采取行动，这个节点将执行这个行动。

我们也需要定义一些边（edges）。有些边可能是条件边。他们作为条件的理由是：以节点输出为基础，可以采取几种路径的一种。直到节点运行前路径被采用是未知的（由大模型决定）。

1. 条件边:在agent被调用之后，我们应该如下两种场景之一：a.如果agent说采取行动，那么函数应该被工具调用。b.如果agent说结束了，那么流程应该结束。 
2. 普通边: 在工具被调用之后，它只会返回结果给agent，由agent决定下一步需要采取什么行动。

让我们定义这些节点，也定义一个函数来决定如何选择条件边。

**STREAMING**

我们定义每一个节点作为异步(async)函数



小注释：手工回调传输 Manual Callback Propagation

> Note that in `call_model(state: State, config: RunnableConfig):` below, we a) accept the [RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig) in the node and b) pass this in as the second arg for `llm.ainvoke(..., config)`. This is optional for python 3.11 and later.
>
> 注意下面的`call_model(state: State, config: RunnableConfig):` 我们 a）在节点中接收一个[RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig)，b）将其作为` llm.ainvoke的第二个参数传入(...，配置)`。对于python 3.11和更高版本，这是可选的。

```python
from typing import Literal

from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, START, StateGraph


# Define the function that determines whether to continue or not
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    else:
        return "tools"


# Define the function that calls the model
async def call_model(state: State, config: RunnableConfig):
    messages = state["messages"]
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    response = await model.ainvoke(messages, config)
    # We return a list, because this will get added to the existing list
    return {"messages": response}
```

**API 参考:** [RunnableConfig](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html) | [END](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.END) | [START](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START) | [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)



#### 定义graph 
[参考：Define the graph](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#define-the-graph)

我们现在将他们组合起来并且定义graph！

```python
# Define a new graph
workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Next we pass in the path map - all the nodes this edge could go to
    ["tools", END],
)

workflow.add_edge("tools", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
```
查看图的流程
```python
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))
```
<img src=".\images\steam_llm_graph.png" alt="可视化graph流程" style="zoom:78%;" />

#### 大模型流式输出Tokens
[参考文档：Streaming LLM Tokens](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#streaming-llm-tokens)



你可以访问大模型的输出tokens，因为他们是由每一个节点输出的。在这个例子中仅仅只有"agent"节点生产出大模型的tokens。为了能正常运行，你必须使用支持流式输出的大模型，并且在构造LLM的时候就设置这种模式（例如：ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)）。

```python
from langchain_core.messages import AIMessageChunk, HumanMessage

inputs = [HumanMessage(content="what is the weather in sf")]
first = True
async for msg, metadata in app.astream({"messages": inputs}, stream_mode="messages"):
    if msg.content and not isinstance(msg, HumanMessage):
        print(msg.content, end="|", flush=True)

    if isinstance(msg, AIMessageChunk):
        if first:
            gathered = msg
            first = False
        else:
            gathered = gathered + msg

        if msg.tool_call_chunks:
            print(gathered.tool_calls)
```
```text
[{'name': 'search', 'args': {}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {'query': ''}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {'query': 'weather'}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {'query': 'weather in'}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {'query': 'weather in San'}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
[{'name': 'search', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_lfwgOci165GXplBjSDBeD4sE', 'type': 'tool_call'}]
["Cloudy with a chance of hail."]|The| weather| in| San| Francisco| is| currently| cloudy| with| a| chance| of| hail|.|
```

**API 参考:** [AIMessageChunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html) | [HumanMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.human.HumanMessage.html)



 <a id='steam7'></a>

### 如何在工具内流式输出数据
[参考文档:How to stream data from within a tool](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/#how-to-stream-data-from-within-a-tool)

#### 前提

这个教程假设你已经熟悉如下内容：

- [Streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [Chat Models](https://python.langchain.com/docs/concepts/#chat-models/)
- [Tools](https://python.langchain.com/docs/concepts/#tools)
- [RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig)
- [RunnableInterface](https://python.langchain.com/docs/concepts/#runnable-interface)

如何你的graph涉及调用大模型（或者像是其他graph一样的LangChain`Runnable` 对象，`LCEL` 链，或检索器）的工具。你可能想要在工具执行期间先显示出部分结果，尤其是工具运行时间较长的情况下。

一个通用的场景是由工具LLM的工具流式输出大模型的tokens，尽管这适用于任何使用Runnable对象的情况。

这个教程展示了如何在工具内用`astream` API 和`stream_mode="messages"`流式输出数据，并且还有更细颗粒度的`astream_events` API。`astream` API估计能满足大部分应用场景。

####  准备

首先，初始化依赖包和设置API key。

```python
%%capture --no-stderr
%pip install -U langgraph langchain-openai
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```
设置 LangSmith 来进行 LangGraph 开发

> Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started here.



#### 定义graph

在本指南中，我们将使用一个预构建的ReAct Agent。

**在PYTHON<=3.10 中ASYNC** 

>Any Langchain `RunnableLambda`, a `RunnableGenerator`, or `Tool` that invokes other runnables and is running async in python<=3.10, will have to propagate callbacks to child objects **manually**. This is because LangChain cannot automatically propagate callbacks to child objects in this case. This is a common reason why you may fail to see events being emitted from custom runnables or tools.

```python
from langchain_core.callbacks import Callbacks
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


@tool
async def get_items(
    place: str,
    callbacks: Callbacks,  # <--- Manually accept callbacks (needed for Python <= 3.10)
) -> str:
    """Use this tool to look up which items are in the given place."""
    # Attention when using async, you should be invoking the LLM using ainvoke!
    # If you fail to do so, streaming will not WORK.
    return await llm.ainvoke(
        [
            {
                "role": "user",
                "content": f"Can you tell me what kind of items i might find in the following place: '{place}'. "
                "List at least 3 such items separating them by a comma. And include a brief description of each item..",
            }
        ],
        {"callbacks": callbacks},
    )


llm = ChatOpenAI(model_name="gpt-4o")
tools = [get_items]
agent = create_react_agent(llm, tools=tools)
```

**API 参考:** [HumanMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.human.HumanMessage.html) | [tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) | [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) | [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)

#### 使用 stream_mode="messages"
[参考文档：Using stream_mode="messages"](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/#using-stream_modemessages)



如果在你的节点（node）内部没有复杂的LCEL逻辑（或者在你的LCEL链内不需要非常系颗粒度的步骤） ，那么使用`stream_mode="messages"`将是一个很好的选择。 

```python
final_message = ""
async for msg, metadata in agent.astream(
    {"messages": [("human", "what items are on the shelf?")]}, stream_mode="messages"
):
    # Stream all messages from the tool node
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata["langgraph_node"] == "tools"
        and not msg.name
    ):
        print(msg.content, end="|", flush=True)
    # Final message should come from our agent
    if msg.content and metadata["langgraph_node"] == "agent":
        final_message += msg.content
```

####  使用 stream events API

[参考文档：Using stream events API](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/#using-stream-events-api)

为了简单，`get_items` 工具内部没有任何复杂的LCEL逻辑——仅调用大模型。



然而，如果工具太复杂（例如，在工具内使用RAG链），并且你想看清楚链内部更细颗粒度的事件(event)，那你可以使用`astream events` API.



下面的例子仅仅展示如何调用API。



**警告：使用异步的`async astream events` API**

> 你通常应该使用`async`代码（例如：使用`ainvoke`来调用大模型）才能充分发挥`astream events` API。

```python
from langchain_core.messages import HumanMessage

async for event in agent.astream_events(
    {"messages": [{"role": "user", "content": "what's in the bedroom."}]}, version="v2"
):
    if (
        event["event"] == "on_chat_model_stream"
        and event["metadata"].get("langgraph_node") == "tools"
    ):
        print(event["data"]["chunk"].content, end="|", flush=True)
```



```tex
|In| a| bedroom|,| you| might| find| the| following| items|:

|1|.| **|Bed|**|:| The| central| piece| of| furniture| in| a| bedroom|,| typically| consisting| of| a| mattress| on| a| frame|,| where| people| sleep|.| It| often| includes| bedding| such| as| sheets|,| blankets|,| and| pillows| for| comfort|.

|2|.| **|Ward|robe|**|:| A| large|,| tall| cupboard| or| fre|estanding| piece| of| furniture| used| for| storing| clothes|.| It| may| have| hanging| space|,| shelves|,| and| sometimes| drawers| for| organizing| garments| and| accessories|.

|3|.| **|Night|stand|**|:| A| small| table| or| cabinet| placed| beside| the| bed|,| used| for| holding| items| like| a| lamp|,| alarm| clock|,| books|,| or| personal| belongings| that| might| be| needed| during| the| night| or| early| morning|.||
```

**API 参考:** [HumanMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.human.HumanMessage.html)