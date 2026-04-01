"""
llm_examples.py
---------------
Practical examples of working with LLMs.

Requires:
    pip install openai anthropic tiktoken

Set environment variables before running:
    OPENAI_API_KEY    - for OpenAI examples
    ANTHROPIC_API_KEY - for Anthropic examples
"""

import os
import textwrap


# ---------------------------------------------------------------------------
# Token counting (no API key needed)
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the number of tokens in *text* for the given model."""
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def demo_token_counting() -> None:
    sample = "Large Language Models are fascinating pieces of technology!"
    n = count_tokens(sample)
    print(f"Text   : {sample!r}")
    print(f"Tokens : {n}")
    print(f"Chars  : {len(sample)}")
    print(f"Ratio  : {len(sample)/n:.1f} chars/token\n")


# ---------------------------------------------------------------------------
# OpenAI chat completion
# ---------------------------------------------------------------------------

def openai_chat(prompt: str, system: str = "You are a helpful assistant.",
                model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """Send a chat message to OpenAI and return the reply."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def demo_openai() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping OpenAI demo.\n")
        return

    prompt = "In two sentences, explain what a transformer neural network is."
    print("=== OpenAI Demo ===")
    print(f"Prompt: {prompt}")
    reply = openai_chat(prompt)
    print(f"Reply :\n{textwrap.indent(reply, '  ')}\n")


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

def claude_chat(prompt: str, model: str = "claude-3-5-haiku-20241022",
                max_tokens: int = 512) -> str:
    """Send a message to Claude and return the reply."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def demo_claude() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping Claude demo.\n")
        return

    prompt = "In two sentences, explain what a transformer neural network is."
    print("=== Anthropic Claude Demo ===")
    print(f"Prompt: {prompt}")
    reply = claude_chat(prompt)
    print(f"Reply :\n{textwrap.indent(reply, '  ')}\n")


# ---------------------------------------------------------------------------
# Prompt engineering examples
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = """\
Classify the sentiment of the following review as Positive, Negative, or Neutral.

Review: "The battery life is okay but the screen is disappointingly dim."
Sentiment:"""

FEW_SHOT_PROMPT = """\
Classify the sentiment of each review as Positive, Negative, or Neutral.

Review: "Absolutely love this product!" → Positive
Review: "Broke after one week. Very disappointed." → Negative
Review: "It does what it says on the box, nothing more." → Neutral
Review: "The battery life is okay but the screen is disappointingly dim." →"""

COT_PROMPT = """\
Think step by step, then answer.

Question: If a train travels at 120 km/h and needs to cover 300 km, how many \
minutes will the journey take?"""


def demo_prompting_styles() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping prompt engineering demo.\n")
        return

    print("=== Zero-shot Sentiment ===")
    print(openai_chat(ZERO_SHOT_PROMPT, temperature=0), "\n")

    print("=== Few-shot Sentiment ===")
    print(openai_chat(FEW_SHOT_PROMPT, temperature=0), "\n")

    print("=== Chain-of-Thought Math ===")
    print(openai_chat(COT_PROMPT, temperature=0), "\n")


# ---------------------------------------------------------------------------
# Simple in-memory vector similarity (no external DB needed)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def demo_embeddings() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping embeddings demo.\n")
        return

    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed today.",
    ]

    print("=== Embedding Similarity ===")
    embeddings = [get_embedding(s) for s in sentences]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}  |  {sentences[i]!r} ↔ {sentences[j]!r}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LLM Fun — Examples ===\n")
    demo_token_counting()
    demo_openai()
    demo_claude()
    demo_prompting_styles()
    demo_embeddings()
