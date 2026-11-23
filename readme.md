# Introduction to LangChain and LangGraph

**Author:** Maryam Berijanian  
**Platform:** Google Colab
**Last Updated:** November 2026

> **New to AI Agents, LangChain, or LangGraph?** Start with our 2-minute visual primer: **[`intro_crash_course.md`](intro_crash_course.md)**.

---

## Overview

This two-part hands-on workshop introduces students, researchers, and developers to building AI applications and autonomous agents using **LangChain** and **LangGraph**.

**Features**:
* **🎛️ Personalized Colab Form Widgets (`#@param`)**: Users can customize inputs.
* **🧩 Hinge-Line Micro-Puzzles (`TODO` + `assert`)**: Targeted 1-to-3 line coding challenges that are verified automatically by `assert` checks and include **💡 Solution** cells.
* **🔴 2-Minute Red-Team Challenges**: Timed adversarial exercises where users can test prompt-injection and router-bypass edge cases.
* **🎓 Capstones**: Part 1 concludes with a **Live ArXiv Study-Guide & Flashcard Extractor**, and Part 2 with a **University Course Survival Copilot** combining RAG, grade math, and Human-in-the-Loop email approval.

---

## Learning Objectives

By completing the notebook, users will be able to:
1. Distinguish between stateless LLM calls, sequential prompt chains, declarative **LCEL** pipelines, and stateful **LangGraph** workflows.
2. Extract typed Python objects from unstructured text.
3. Implement dynamic control flow using **conditional edges** (routing) and **cyclic graphs** (iterative loops).
4. Build custom external tools connecting LLMs to live REST APIs (Open-Meteo weather, Wikipedia, and ArXiv research papers).
5. Ground LLM responses in private documents using **Retrieval-Augmented Generation (RAG)** with Hugging Face embeddings and **FAISS**.
6. Add multi-turn conversational memory and **Human-in-the-Loop (HITL)** safety gates.

---

## Prerequisites & Quick Setup (Google Colab)

### Prerequisites
* Basic familiarity with Python.
* A standard Google account with access to [Google Colab](https://colab.research.google.com/).  
(Tip: Students and educators can claim **Google Colab Pro for free** at [colab.research.google.com/signup](https://colab.research.google.com/signup).)

### Step 1: Save Your Own Copy
Before editing either workshop notebook in Google Colab, click **`File -> Save a copy in Drive`**. Both parts include setup cells, so Part 2 runs even if your Colab runtime resets.

### Step 2: Configure One Free Model Provider

| Provider | Cost | Colab Runtime | Native Tool Calling | Recommended Use |
| :--- | :--- | :--- | :--- | :--- |
| **1. `gemini` (Primary)** | **Free Tier** | Standard CPU | ✅ Full Support | **Best for Part 1 & Part 2** |
| **2. `huggingface` (Local)** | **Free** | Free **T4 GPU** | ⚠️ Limited (Best for Part 1 & RAG) | **Zero-API-Key Local Fallback** |
| **3. `openai` (Optional)** | Paid API | Standard CPU | ✅ Full Support | Optional alternative (`gpt-4o-mini`) |

#### Setting Up Option 1: Gemini Free Tier (Recommended)
1. Open Google Colab and click the **🔑 (Secrets)** icon in the left sidebar.
2. Click **"Import key from Google AI Studio"** (or generate a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).
3. Name the secret **`GEMINI_API_KEY`** and toggle **Notebook access** to **ON**.

#### Setting Up Option 2: Zero-Key Local Hugging Face Model (`Qwen/Qwen2.5-3B-Instruct`)
1. Go to **`Runtime -> Change runtime type -> T4 GPU`**.
2. Set `MODEL_PROVIDER = "huggingface"` in the setup cell.  
*(Note: Part 1 runs 100% locally on `huggingface`; Part 2's multi-tool ReAct agent and Capstone require native `.bind_tools()` via `gemini` or `openai`.)*

---

## Curriculum

### Part 1: Foundations, Structured Outputs, Routing & Cyclic Reflection
**File:** `Workshop_Part1_Foundations_and_Routing.md`  
*Focus: Typed LCEL pipelines and cyclic, self-correcting state graphs.*

**Architectures:**

1. **Prompt Chain (Sec 3):** $\text{Input} \rightarrow \text{Prompt} \rightarrow \text{LLM} \rightarrow \text{Parser} \rightarrow \text{Output}$
2. **Conditional Router (Sec 5):** $\text{START} \rightarrow \text{Classify} \rightarrow \text{Plan} \rightarrow \{\text{Research} \mid \text{Skip}\} \rightarrow \text{Write} \rightarrow \text{END}$
3. **Cyclic Reflection (Sec 6):** $\text{START} \rightarrow \text{Generator} \rightleftarrows \text{Rubric Critic} \rightarrow \text{END}$

### Part 2: External Tools, Agentic RAG, Memory, HITL & Capstone
**File:** `Workshop_Part2_Agents_RAG_and_HITL.md`  
*Focus: Connecting LangGraph agents to live APIs, vector databases, persistent thread memory, and human approval workflows.*

**Architectures:**

1. **ReAct + RAG + HITL Agent (Sec 4):** $\text{START} \rightarrow \text{Copilot LLM} \rightleftarrows \text{ToolNode (Syllabus FAISS + Grade Math + ArXiv API + HITL Email Interrupt)} \rightarrow \text{END}$

---

## Troubleshooting

* **`Missing GEMINI_API_KEY` Error:** Toggle the **Notebook access** switch ON next to `GEMINI_API_KEY` in the left Colab `🔑 Secrets` sidebar.
* **Agent Ignores a Custom `@tool`:** Inspect `tool.description` and `tool.args`. LLMs rely strictly on the Python docstring and parameter names to decide when to invoke a tool (Puzzle 2.1 and Capstone `TODO #1a`).
* **Graph Loops Forever:** Always pair cyclic edges (`Reflection` in Part 1) with an explicit `iteration >= max_iters` guard inside the conditional routing function.  

* **Rate Limit:** With the default **`gemini` (`gemini-2.5-flash`)** free-tier configuration in Google AI Studio, each personal API key is limited to approximately 10–15 requests per minute. **Please use your quota responsibly:**

1. **Do not put LLM calls inside tight loops** or repeatedly click *Run All* while a cell is still executing.
2. **Multi-step workflows make multiple calls:** A single ReAct agent run, cyclic reflection loop, or multi-agent supervisor run typically makes **2 to 4 API calls** under the hood.
3. **If you hit a `429 RESOURCE_EXHAUSTED` error:**, wait **30–60 seconds** before re-running the cell.


---

## References & Further Learning

1. **[LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)** — Free official LangGraph courses (<https://github.com/langchain-ai/langchain-academy>)
2. **[LangGraph Documentation: Persistence & Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)**
3. **[LangChain Conceptual Guides (LCEL, Structured Outputs, Tools, RAG)](https://python.langchain.com/docs/concepts/)**
4. **[A Hands-On Guide to Building Intelligent Systems](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/preview?tab=t.0)** — Antonio Gulli
5. **[Prof. Mohammad Ghassemi — AI Agents Lectures](https://www.youtube.com/@ghassemi)**