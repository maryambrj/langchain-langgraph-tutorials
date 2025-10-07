<a href="https://colab.research.google.com/drive/1WLSjEqJAkvs4BXLG9V8eYJonBIzas4Yf#scrollTo=NbqLXtfPIzJR" target="_blank" rel="noopener">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

# LangChain + LangGraph Intro

A hands-on starter repo for building **agentic LLM workflows** with **LangChain** and **LangGraph** entirely on free/open tooling (Hugging Face models, FAISS, Wikipedia, Open-Meteo). It walks from “single LLM call” to **routing**, **tools**, **reflection**, **RAG**, **multi-agent collaboration**, and **human-in-the-loop** (HITL) patterns.

## Contents

* Run small open models (TinyLlama / Qwen 1.5B / Phi-3 mini) locally or in Colab
* Compose chains with LCEL (`prompt | model | parser`)
* Route with LangGraph (conditional edges, loops)
* Add tools (Open-Meteo weather, Wikipedia) with `@tool`
* Reflect & revise (self & external critique)
* RAG with FAISS + HuggingFaceEmbeddings
* Multi-agent “Researcher → Analyst” pattern
* HITL interruptions & simple tool router
* (Bonus) Graph visualization

---

## Quickstart

1. **Open the notebook**
   Upload or open `LangChain-Intro.ipynb` in Google Colab.

2. **Set a Hugging Face token (free)**
   Create a token at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   In Colab, click the **🔑 (Secrets)** icon → add a secret named **`HF_TOKEN`**.

3. **Install deps** (first code cell already included in the notebook)

```bash
!pip install -q -U \
  "requests==2.32.4" "numpy==2.0.2" \
  "transformers==4.41.2" "accelerate>=0.29,<1" \
  "langchain>=0.3,<0.4" "langchain-community>=0.3,<0.4" \
  "langgraph==0.2.*" "langchain-huggingface>=0.1,<0.4" \
  "sentence-transformers>=2.6,<3" "wikipedia" \
  "sentencepiece" "faiss-cpu" "tiktoken" "ipywidgets"
```

4. **Run cells top→bottom**
   The notebook prints which model it loaded and whether it’s using GPU.

---

## Local setup (optional)

* **Python:** 3.10–3.11 recommended
* **Create venv & install:**

  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -U requests==2.32.4 numpy==2.0.2 \
    transformers==4.41.2 "accelerate>=0.29,<1" \
    "langchain>=0.3,<0.4" "langchain-community>=0.3,<0.4" \
    langgraph==0.2.* "langchain-huggingface>=0.1,<0.4" \
    "sentence-transformers>=2.6,<3" wikipedia sentencepiece faiss-cpu tiktoken ipywidgets
  ```
* **HF token:** export `HF_TOKEN` in your shell or `.env`.

## Models

Default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fast, small).
Alternatively, change `MODEL_NAME` env or variable to a different model.

---

## Walkthrough

### 1) Setup

* Installs Transformers/Accelerate/LangChain/LangGraph, etc.
* Loads a small chat model via `transformers.pipeline`.
* Wraps it with `HuggingFacePipeline` and `ChatHuggingFace`.

### 2) Basic LLM Call

* Minimal “chat.invoke” and an LCEL chain:

  ```python
  prompt | chat | StrOutputParser()
  ```
* Shows how to keep generations concise and deterministic-ish (`temperature=0.3`).

### 3) **Routing vs. Chaining** (LangGraph)

* **Pure chain:** `plan → research → write`
* **Routed graph:** `classify → plan → (skip write | research → write)`
* Includes normalizers (`normalize_plan/facts/answer`) so outputs are tidy and bounded.

### 4) **Tool Use**

* Create tools with `@tool` (Pydantic args supported).
* Examples:

  * **Open-Meteo**: `get_current_temperature(latitude, longitude)`
  * **Wikipedia**: `search_wikipedia(query)`
* Demonstrates `convert_to_openai_function` for function-calling metadata.
* Note: Tiny HF chat models won’t auto-call tools; LangGraph/agents handle invocation.

### 5) **Reflection**

* **Self-reflection**: draft → critique → revise
* **External reflection**: draft → external critic → revise
* Clean prompts and “only return X” constraints to keep outputs scoped.

### 6) **RAG (Retrieval-Augmented Generation)**

* **FAISS** as vector store + **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`)
* Small in-memory doc set → similarity search → context → generate
* Behavior when the context doesn’t cover the query (safely says it can’t answer).

### 7) **Multi-Agent Collaboration**

* “Researcher” generates bullets → “Analyst” summarizes into 1 sentence.
* Illustrates role-prompting with simple functions.

### 8) **Human-in-the-Loop (HITL)**

* LangGraph with a **Message reducer**, **MemorySaver** checkpoints, and an **interrupt** before actions.
* A tiny **router** picks a tool from user text (weather/Wikipedia/search fallback).
* Shows how a human can edit the planned tool input before execution.

### Bonus) **Visualization**

* Export the compiled graph via Mermaid/Graphviz → PNG.

---

## Configuration

* **Environment variable**: `HF_TOKEN` – required for gated models/rate-limits on the Hub.  
* **GPU**: Auto-detect via `torch.cuda.is_available()`. On CPU, keep models small.

<!--## Troubleshooting)-->
<!--CUDA OOM / slow runs: switch to TinyLlama; lower `max_new_tokens`; use CPU if GPU is flaky.-->
<!--HF auth errors: ensure `HF_TOKEN` is set in Colab Secrets (exact name) and restart runtime.-->
<!--Wikipedia errors: the `wikipedia` Python lib can raise disambiguation errors; the tool handles most, but try a more specific query.-->
<!--Version conflicts: the notebook pins versions—use the provided `pip` cell unmodified.-->

## Security & Costs

* All examples use **free** APIs or local inference.
* The weather tool calls **Open-Meteo** (no key required).
* Do **not** store secrets in the notebook; use Colab Secrets or local env vars.

---

## References
1. [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph) – curated courses & tutorials:    
  GitHub: <https://github.com/langchain-ai/langchain-academy.git>

2. [LangGraph Cookbook](https://github.com/abhishekmaroon5/langgraph-cookbook)

3. [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)

4. [Interrupt](https://interrupt.langchain.com/) – the AI Agent Conference by LangChain  

5. [A Hands-On Guide to Building Intelligent Systems](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/preview?tab=t.0), Antonio Gulli

<!-- 6. [Prof. Ghassemi Lectures and Tutorials](https://www.youtube.com/@ghassemi), AI Agents lectures -->
---

## Citation

If this helped your class or workshop, please cite:

```
Maryam Berijanian. (2025).
LangChain + LangGraph Intro Workshop. GitHub repository.
```
---

## Contributing

PRs to improve examples, add agents/tools, or extend the RAG section are welcome! Please:

* Pin versions and include short comments in code cells.
* Add small, deterministic tests where feasible.

