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

