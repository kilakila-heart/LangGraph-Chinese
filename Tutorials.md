# Tutorials[操作指南](https://langchain-ai.github.io/langgraph/tutorials/#tutorials)

欢迎来到LangGraph指南！这些笔记通过创建各种大语言agent和应用介绍了LangGraph。

## 快速入门[点击进入](https://langchain-ai.github.io/langgraph/tutorials/#quick-start)

通过一个完整的快速入门学习LangGraph的基础，你将可以从零开始构建一个agent.

- [快速入门](QuickStart.md)

## 用户案例[¶](https://langchain-ai.github.io/langgraph/tutorials/#use-cases)

从特定场景的设计和相同的设计模式的Graph示例中学习。

#### 聊天机器人[¶](https://langchain-ai.github.io/langgraph/tutorials/#chatbots)

- [智能客服](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/): 创建一个智能客服机器人管理航班、酒店预订、汽车租赁和其他任务。
- [根据用户需求生成Prompt](https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/): 建立一个信息收集聊天机器人
- [代码助手](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/): 创建一个diamond分析和生成助手

#### 多agent系统[¶](https://langchain-ai.github.io/langgraph/tutorials/#multi-agent-systems)

- [合并](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/): 使两个代理能够协作完成一项任务

- [管理](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/): 使用LLM来编排和委派给各个agent

- [Hierarchical Teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/): 编排嵌套的agent团队来解决问题

## RAG[¶](https://langchain-ai.github.io/langgraph/tutorials/#rag)

- [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/): Use an agent to figure out how to retrieve the most relevant information before using the retrieved information to answer the user's question.

- Adaptive RAG

  : Adaptive RAG is a strategy for RAG that unites (1) query analysis with (2) active / self-corrective RAG. Implementation of:

   

  https://arxiv.org/abs/2403.14403

  - For a version that uses a local LLM: [Adaptive RAG using local LLMs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)

- Corrective RAG

  : Uses an LLM to grade the quality of the retrieved information from the given source, and if the quality is low, it will try to retrieve the information from another source. Implementation of:

   

  https://arxiv.org/pdf/2401.15884.pdf

  - For a version that uses a local LLM: [Corrective RAG using local LLMs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/)

- Self-RAG

  : Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations. Implementation of

   

  https://arxiv.org/abs/2310.11511

  .

  - For a version that uses a local LLM: [Self-RAG using local LLMs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag_local/)

- [SQL Agent](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/): Build a SQL agent that can answer questions about a SQL database.

## Agent 结构
[参考文档：Agent Architectures](https://langchain-ai.github.io/langgraph/tutorials/#agent-architectures)

### 多Agent系统
[参考文档：Multi-Agent Systems](https://langchain-ai.github.io/langgraph/tutorials/#multi-agent-systems)

- [Network](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/): Enable two or more agents to collaborate on a task
- [Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/): Use an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/): Orchestrate nested teams of agents to solve problems

### 计划Agent
[参考文档：Planning Agents](https://langchain-ai.github.io/langgraph/tutorials/#planning-agents)

- [计划和执行](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/): 实现了一个基础的计划和执行的agent
- [Reasoning without Observation](https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/): Reduce re-planning by saving observations as variables
- [LLMCompiler](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/): Stream and eagerly execute a DAG of tasks from a planner

### Reflection & Critique[¶](https://langchain-ai.github.io/langgraph/tutorials/#reflection-critique)

- [Basic Reflection](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/): Prompt the agent to reflect on and revise its outputs
- [Reflexion](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/): Critique missing and superfluous details to guide next steps
- [Language Agent Tree Search](https://langchain-ai.github.io/langgraph/tutorials/lats/lats/): Use reflection and rewards to drive a tree search over agents
- [Self-Discover Agent](https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover/): Analyze an agent that learns about its own capabilities

## Evaluation[¶](https://langchain-ai.github.io/langgraph/tutorials/#evaluation)

- [Agent-based](https://langchain-ai.github.io/langgraph/tutorials/chatbot-simulation-evaluation/agent-simulation-evaluation/): Evaluate chatbots via simulated user interactions
- [In LangSmith](https://langchain-ai.github.io/langgraph/tutorials/chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation/): Evaluate chatbots in LangSmith over a dialog dataset