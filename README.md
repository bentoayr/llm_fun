# llm_fun

Random stuff on Large Language Models (LLMs) — notes, examples, and experiments.

---

## What is a Large Language Model?

A **Large Language Model (LLM)** is a type of deep learning model trained on massive amounts of text data to understand and generate human language. LLMs are built on the **Transformer** architecture (introduced in the 2017 paper *"Attention Is All You Need"*) and are capable of tasks like:

- Text generation and completion
- Summarization
- Translation
- Question answering
- Code generation
- Reasoning and math

Popular LLMs include **GPT-4**, **Claude**, **Gemini**, **LLaMA**, **Mistral**, and **Falcon**.

---

## Key Concepts

### Tokens
LLMs don't read raw characters — they read **tokens**, which are chunks of text (roughly 3–4 characters on average in English). A rule of thumb: **1,000 tokens ≈ 750 words**.

### Context Window
The **context window** is the maximum number of tokens an LLM can process at once (both input and output). Modern models range from 4K to over 1M tokens.

| Model          | Context Window |
|----------------|---------------|
| GPT-3.5 Turbo  | 16K tokens    |
| GPT-4o         | 128K tokens   |
| Claude 3.5     | 200K tokens   |
| Gemini 1.5 Pro | 1M tokens     |

### Temperature
**Temperature** controls randomness in outputs:
- `0.0` → deterministic, always picks the most likely token
- `1.0` → more creative/varied
- `>1.0` → increasingly random

### Top-p (Nucleus Sampling)
**Top-p** sampling restricts token choices to the smallest set whose cumulative probability exceeds `p`. Typically set to `0.9` or `0.95`.

### Embeddings
**Embeddings** are dense vector representations of text. Similar meaning → similar vectors. Used for semantic search, clustering, and RAG pipelines.

---

## Prompt Engineering Tips

1. **Be specific** — vague prompts give vague answers.
2. **Use examples** (few-shot prompting) — show the model what format you want.
3. **Chain of Thought (CoT)** — ask the model to "think step by step" to improve reasoning.
4. **System prompts** — set the model's persona and constraints upfront.
5. **Iterate** — prompt engineering is experimental; refine based on outputs.

### Example: Zero-shot vs Few-shot

**Zero-shot:**
```
Classify this review as positive or negative: "The food was amazing!"
```

**Few-shot:**
```
Classify these reviews as positive or negative:

Review: "Loved it!" → positive
Review: "Terrible experience." → negative
Review: "The food was amazing!" →
```

---

## Retrieval-Augmented Generation (RAG)

**RAG** combines LLMs with a retrieval system to ground responses in real documents:

1. Embed your documents into a vector database (e.g., Chroma, Pinecone, pgvector)
2. Embed the user's query
3. Retrieve the top-k most similar document chunks
4. Pass retrieved chunks + query to the LLM as context

This reduces hallucinations and lets you use up-to-date or proprietary data without fine-tuning.

---

## Fine-tuning vs Prompting

| Approach       | When to use                                        | Cost      |
|----------------|----------------------------------------------------|-----------|
| Prompting      | Most tasks; fast iteration                        | Low       |
| Few-shot       | Need specific output format                       | Low       |
| RAG            | Custom/private knowledge base                     | Medium    |
| Fine-tuning    | Consistent style/behavior; domain-specific tasks  | High      |
| Pre-training   | Entirely new domain (rare)                        | Very High |

---

## Quick Start: Calling an LLM API

### OpenAI (Python)

```python
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in one paragraph."},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Anthropic Claude (Python)

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

message = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain transformers in one paragraph."}
    ],
)

print(message.content[0].text)
```

### Local Models with Ollama

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
ollama run llama3.2 "Explain transformers in one paragraph."
```

---

## Interesting Facts About LLMs

- GPT-3 has **175 billion parameters**; GPT-4's parameter count is undisclosed but estimated in the hundreds of billions.
- Training GPT-3 consumed roughly **1,287 MWh** of electricity — equivalent to flying across the Atlantic ~500 times.
- LLMs can exhibit **emergent abilities** — capabilities that appear suddenly at scale and weren't present in smaller models (e.g., multi-step arithmetic, analogical reasoning).
- **"Hallucination"** is the term for when an LLM confidently generates false information.
- The **"lost in the middle"** problem: LLMs tend to pay less attention to information in the middle of a long context window.
- **RLHF** (Reinforcement Learning from Human Feedback) is the technique behind ChatGPT's conversational alignment — humans rank outputs, and the model is trained to prefer higher-ranked responses.

---

## Useful Resources

- [Attention Is All You Need (original Transformer paper)](https://arxiv.org/abs/1706.03762)
- [OpenAI Docs](https://platform.openai.com/docs)
- [Anthropic Docs](https://docs.anthropic.com)
- [Hugging Face](https://huggingface.co) — open-source models and datasets
- [LangChain](https://www.langchain.com) — framework for building LLM apps
- [LlamaIndex](https://www.llamaindex.ai) — data framework for LLM applications
- [Ollama](https://ollama.com) — run LLMs locally
- [Simon Willison's Blog](https://simonwillison.net) — great LLM commentary and experiments
