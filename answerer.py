import os
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from retriever import retrieve
from router import route_answer_mode


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def build_context(results: Dict) -> str:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "unknown")

        context_parts.append(
            f"[Source {i}] {source} | chunk {chunk_index}\n{doc}"
        )

    return "\n\n".join(context_parts)


def build_style_instructions(answer_style: str, include_example: bool) -> str:
    style_parts = []

    if answer_style == "Plain-language":
        style_parts.append(
            "Answer in a plain-language style. "
            "Explain the idea as clearly and simply as possible for a beginner, "
            "while preserving important technical terms in English when helpful."
        )
    else:
        style_parts.append(
            "Answer in a technical mathematical style. "
            "Use precise definitions and standard terminology."
        )

    if include_example:
        style_parts.append(
            "If possible, include one short simple example or intuitive illustration. "
            "If the example is not directly supported by the retrieved library, clearly label it as a general illustration."
        )

    return " ".join(style_parts)


def answer_question(
    question: str,
    top_k: int = 5,
    answer_style: str = "Technical",
    include_example: bool = False,
) -> Dict[str, Any]:
    results = retrieve(query=question, top_k=top_k)
    context = build_context(results)
    mode, reason = route_answer_mode(results)

    style_instruction = build_style_instructions(answer_style, include_example)

    if mode == "Library-grounded":
        system_prompt = (
            "You are a careful mathematical reading assistant. "
            "Answer mainly from the retrieved library context. "
            "Be explicit about which sources support the answer. "
            "Do not invent citations. "
            "Format mathematical notation using clean LaTeX when appropriate. "
            + style_instruction
        )
    elif mode == "Hybrid":
        system_prompt = (
            "You are a careful mathematical reading assistant. "
            "Use the retrieved library context first, but you may add limited general explanation "
            "if the library context is relevant but incomplete. "
            "Clearly distinguish between library-supported content and general background explanation. "
            "Do not invent citations. "
            "Format mathematical notation using clean LaTeX when appropriate. "
            + style_instruction
        )
    else:
        system_prompt = (
            "You are a careful mathematical reading assistant. "
            "The current library coverage appears insufficient. "
            "Say that clearly. You may still provide a cautious general explanation, "
            "but do not pretend the answer is fully supported by the library. "
            "Do not invent citations. "
            "Format mathematical notation using clean LaTeX when appropriate. "
            + style_instruction
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"""
Answer mode:
{mode}

Routing reason:
{reason}

User question:
{question}

Retrieved context:
{context}

Instructions:
1. Prefer the retrieved context.
2. Be concise but clear.
3. If possible, mention which source(s) support the answer.
4. Do not invent citations.
5. Format mathematical expressions in LaTeX.
6. Use $...$ for inline math and $$...$$ for display math.
7. Keep notation readable for mathematical writing.
8. Clearly distinguish between statements supported by the retrieved library and any general background explanation.
9. When the question asks for a mathematical concept, start with a precise definition before giving further explanation.
10. If the user asks in Chinese, answer in Chinese while preserving important technical terms in English when helpful.
""",
        },
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.2,
    )

    return {
        "mode": mode,
        "reason": reason,
        "answer": response.choices[0].message.content,
        "documents": results.get("documents", [[]])[0],
        "metadatas": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0],
    }


if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    result = answer_question(question, top_k=5)

    print("\n=== Mode ===\n")
    print(result["mode"])

    print("\n=== Routing Reason ===\n")
    print(result["reason"])

    print("\n=== Answer ===\n")
    print(result["answer"])