# Quick Start

在这个完整的快速入门中，我们将用LangGraph构建一个支持各种功能的机器人，它能：
- 通过搜索网页回答常见问题
- 保持在调用中的对话状态（state)
- 传送复杂的查询给人工审查
- 用自定义的状态state控制它的行为
- 回溯和探索其他(可选)的对话路径

我们将一路从一个基本的聊天机器人开始，逐步增加更复杂的功能，引入关键的LangGraph概念。

## 准备

首先，初始化依赖包：

```bash
%%capture --no-stderr
%pip install -U langgraph langsmith langchain_anthropic
```

接着, 设置你的 API keys:


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
ANTHROPIC_API_KEY:  ········
```

设置 [LangSmith](https://smith.langchain.com/)  LangGraph 开发模式（非必须，先不翻译）

Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com/).



## Part 1: 创建一个基础的聊天机器人
我们将用LangGraph创建一个简单的聊天机器人。这个机器人将直接返回用户消息。虽然简单，但是它将阐明创建LangGraph的核心概念。通过该章节最后，你将有一个基本的聊天机器人。

开始创建一个 `StateGraph`.  `StateGraph` 对象定义了我们的聊天机器人作为一个“状态机器”的结构。我们将添加`节点`来表示我们的聊天机器人可以调用大模型（llm）和函数（functions），并添加“边”来指定机器人应该如何在这些函数之间切换（调用）。



```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```

注意

> 当你一开始定义了一个graph就意味着定义了这个graph的状态(state)，这个`状态`包含了graph的schema以及reducer 函数，这些reducer 函数规定了如何将更新内容应用到状态（state）上。在我们的例子中，`State`是带有单独key（messages）的 `TypedDict`，这个  `messages` keys是一个带有 [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages) 的注解（annotated）的reducer 函数，它将告诉LangGraph 追加mesage到已经存在的消息列表上，而不是覆盖。没有注解的`state`键将被每次更新覆盖，存储最近的值，查阅[这个概念说明](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages) 学习更多关于状态state，reducers和其他的基本概念。



所以现在我们的graph清楚了2件事：

1. 我们定义的每一个节点（node）都会接收当前的状态（state）作为输入和修改了状态state值作为输出。
2. 消息（messages)将会追加到当前的列表中，而不是直接覆盖。这是通过预先构建的`Annotated` 语法[`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages)函数。

接下来，添加一个 "`chatbot`" 节点.节点代表工作单元，这是一个典型的常规python函数。

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
```

**注意** `chatbot`节点函数怎样携带当前 `State` 作为输入并返回一个字典dictionary，字典内包含key="messages"是更新过的`messages` 列表。这是所有LangGraph节点函数的基本模式。



在状态（state）中追加大模型返回的消息，无论这个消息是否已经存在在state里面，这就是在我们的这个`add_messages` 函数的功能。



接下来，我们添加一个`entry`（入口）点，这是告诉图（Graph）每次运行的时候**从哪里开始**。



```python
graph_builder.add_edge(START, "chatbot")
```


同样，设置一个`finish`（结束） 点。这是graph**"在节点运行的任何时候，你都可以结束"**的指令。


```python
graph_builder.add_edge("chatbot", END)
```

Finally, we'll want to be able to run our graph. To do so, call "`compile()`" on the graph builder. This creates a "`CompiledGraph`" we can use invoke on our state.
最后，我们想要运行我们的graph，这么做，调用graph builder的"`compile()`" 函数。这就是创建 "`CompiledGraph（编译过的graph）`"我们可以在我们的state上调用。


```python
graph = graph_builder.compile()
```

You can visualize the graph using the `get_graph` method and one of the "draw" methods, like `draw_ascii` or `draw_png`. The `draw` methods each require additional dependencies.

你可以可视化的看到graph的样子，用像是`draw_ascii` 或 `draw_png`库的 `get_graph` 方法和其中一个"draw"方法

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

<img src=".\images\simple_chatbot_flow.png" alt="可视化graph流程" style="zoom:78%;" />
现在开始运行这个聊天机器人吧！

Tip: 你可以在任何时候使用键盘 "quit", "exit", or "q",结束聊天循环。

```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
           
```

小助手提示：LangGraph是一个用大语言模型帮助构建有状态的多agent应用的框架。它提供了创建工作流程和状态机的工具，来协调AI agent 或者大语言模型的交互。langgraph建立在langchain之上，利用其组件并添加了基于grahp的协调能力。LangGraph对开发复杂、有状态的，不仅仅是简单的查询-响应式的交互应用是特别有用。
再见

**恭喜!** 你已经用LangGraph创建了你的第一个聊天机器人。这个机器人能通过聊天用户的输入和大模型生成响应参与基础对话 .你可以在提供的链接中检查调用的 [LangSmith 轨迹](https://smith.langchain.com/public/7527e308-9502-4894-b347-f34385740d5a/r) 。


然而，你可能已经注意到了这个机器人的知识受限于它自己的训练数据.在下一章节，我们将添加网络搜索工具（web search tool）来扩展机器人的知识库并使它更加强大。

下面的是这个章节的完整代码供你参考：
```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

```

## Part 2: 用工具增强聊天机器人

为了处理用户输入消息，我们的聊天机器人不能凭"记忆"回答，我们将整合一个联网搜索工具。
我们的机器人能用这个工具就能发现相关的信息和提供好的回答。
### 前置要求
在开始之前，确保你已经安装好了必要依赖包和准备好API的keys.
首先，用 [Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/)初始化这些依赖包，并设置你的[TAVILY_API_KEY](https://tavily.com/).

```bash
%%capture --no-stderr
%pip install -U tavily-python langchain_community
```



```python
_set_env("TAVILY_API_KEY")
TAVILY_API_KEY:  ········
```

接下来，定义这个工具：
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")
```
```json
[{'url': 'https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141',
  'content': 'Nodes: Nodes are the building blocks of your LangGraph. Each node represents a function or a computation step. You define nodes to perform specific tasks, such as processing input, making ...'},
 {'url': 'https://saksheepatil05.medium.com/demystifying-langgraph-a-beginner-friendly-dive-into-langgraph-concepts-5ffe890ddac0',
  'content': 'Nodes (Tasks): Nodes are like the workstations on the assembly line. Each node performs a specific task on the product. In LangGraph, nodes are Python functions that take the current state, do some work, and return an updated state. Next, we define the nodes, each representing a task in our sandwich-making process.'}]
```

这个返回结果是我们的聊天机器人能回答问题的页面总结。


接下来，我们开始定义我们的Graph。除了我们增加了绑定工具到大模型上，之后都是与第一张（Part 1）相同的操作。如果大模型想使用我们的搜索引擎，这个操作就让大模型知道正确使用JSON格式。

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
```

下面，如果这些工具被调用到的时候，我们需要创建一个函数去真正的运行它们。我们将把这些工具添加到一个新的节点（node）中。

接着，我们将实现一个基本工具节点`BasicToolNode`，用于在状态（state）中检查最近的消息，并且如果消息中（llm 返回的）包含tool_calls ,就能调用这些工具。当然这依赖大模型对工具调用（tool_calling）的支持，目前Anthropic, OpenAI, Google Gemini，大部分大语言模型都能支持。

我们之后将用LangGraph的预构建工具节点来替换它，以加快速度，但是我们首先自己构建是有指导意义的，能让我们更明白其中原理。
```python
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```
通过对节点的添加，我们定义了条件边`conditional_edges`。

Recall that edges route the control flow from one node to the next. Conditional edges usually contain "if" statements to route to different nodes depending on the current graph state. These functions receive the current graph state and return a string or list of strings indicating which node(s) to call next.

回想一下，边（edge）规划了一个节点到下一个节点的控制流程。条件边通常包含`if`表达式，依赖graph的当前状态`state`发送到不同的节点。这些函数接收当前graph的state，并返回一个字符串或字符串集合来指明下一个应该调用的节点。



下面，调用定义一个名为route_tools的路由函数，它用来检查聊天机器人输出的`tool_calls `。通过调用add_conditional_edges为graph提供此函数，它告诉graph每当聊天机器人节点完成检查该函数来查看下一步去。



如果调用工具存在，这个条件将会发送到工具，否则结束。



之后，为了更简单明了，我们将用预发布tools_condition来替换，但是首先自己来实现将会更加透彻。

```python
from typing import Literal


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

**注意** 条件边从一个单独的节点开始。这告诉graph，每当聊天机器人节点运行时，如果调用工具，将会去调用’tools‘，如果它直接响应，则结束循环。



像预建的“tools_condition”一样，如果没有工具调用，我们的函数将返回“END”字符串。当graph 识别到了`END`,就表示它没有更多的任务要完成并停止执行。因为条件能返回`END`，所以这时我们不需要明确的设置一个`finish_point`，我们的graph已经有了结束的方式。

Let's visualize the graph we've built. The following function has some additional dependencies to run that are unimportant for this tutorial.

为了可直观的看到我们已经创建的graph，下面的函数有一些增加的依赖来运行，这些依赖对本教程来说并不是很重要。

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

<img src=".\images\chatbot_tools_flow.png" alt="带有tools的graph流程" style="zoom:80%;" align="left" />

现在我们可以问一些不在与训练过的数据的相关问题了。

```python
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

```
Assistant: [{'text': "To provide you with accurate and up-to-date information about LangGraph, I'll need to search for the latest details. Let me do that for you.", 'type': 'text'}, {'id': 'toolu_01Q588CszHaSvvP2MxRq9zRD', 'input': {'query': 'LangGraph AI tool information'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Assistant: [{"url": "https://www.langchain.com/langgraph", "content": "LangGraph sets the foundation for how we can build and scale AI workloads \u2014 from conversational agents, complex task automation, to custom LLM-backed experiences that 'just work'. The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution ..."}, {"url": "https://github.com/langchain-ai/langgraph", "content": "Overview. LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures ..."}]
Assistant: Based on the search results, I can provide you with information about LangGraph:

1. Purpose:
   LangGraph is a library designed for building stateful, multi-actor applications with Large Language Models (LLMs). It's particularly useful for creating agent and multi-agent workflows.

2. Developer:
   LangGraph is developed by LangChain, a company known for its tools and frameworks in the AI and LLM space.

3. Key Features:
   - Cycles: LangGraph allows the definition of flows that involve cycles, which is essential for most agentic architectures.
   - Controllability: It offers enhanced control over the application flow.
   - Persistence: The library provides ways to maintain state and persistence in LLM-based applications.

4. Use Cases:
   LangGraph can be used for various applications, including:
   - Conversational agents
   - Complex task automation
   - Custom LLM-backed experiences

5. Integration:
   LangGraph works in conjunction with LangSmith, another tool by LangChain, to provide an out-of-the-box solution for building complex, production-ready features with LLMs.

6. Significance:
   LangGraph is described as setting the foundation for building and scaling AI workloads. It's positioned as a key tool in the next chapter of LLM-based application development, particularly in the realm of agentic AI.

7. Availability:
   LangGraph is open-source and available on GitHub, which suggests that developers can access and contribute to its codebase.

8. Comparison to Other Frameworks:
   LangGraph is noted to offer unique benefits compared to other LLM frameworks, particularly in its ability to handle cycles, provide controllability, and maintain persistence.

LangGraph appears to be a significant tool in the evolving landscape of LLM-based application development, offering developers new ways to create more complex, stateful, and interactive AI systems.
Goodbye!
```

**恭喜!** 你已经用langgraph创建了一个可以在需要时利用搜索引擎检索更新信息的对话功能的agent了。现在它能处理更广泛的问题输入了，为了检查agent的整个步骤，你仅仅需要查看[LangSmith 轨迹](https://smith.langchain.com/public/4fbd7636-25af-4638-9587-5a02fdbb0172/r).



我们的聊天机器人任然不能记住它自己过往的交互信息，限制了它自己的连贯性，多轮对话。在下一章，我们将通过添加**记忆**功能来解决这一问题。



我们在这一章节已经创建了graph完整代码如下所示，用与构建的 [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)替换我们自己的`BasicToolNode` ，并且用预构建的[tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)替换我们自己的`route_tools` 。

### 完整代码：

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
# 任何时候一个工具被调用，我们将返回给聊天机器人来决定下一个步骤
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()
```

## Part 3: 给聊天机器人添加记忆功能
TODO 翻译工作待完成，可以先看官方手册。
[第三章、给机器人添加记忆功能](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot)


