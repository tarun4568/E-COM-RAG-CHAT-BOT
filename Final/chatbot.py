"""
chatbot.py – E-Commerce RAG Chatbot
────────────────────────────────────────────────────────────────────────
Retrieval-Augmented Generation chatbot for ShopEasy E-Commerce support.

Features:
  • Context-aware multi-turn conversation (remembers chat history)
  • Top-k FAISS retrieval per query
  • Strict grounding: answers only from provided documents
  • Graceful fallback when answer is not in the knowledge base

Usage:
    python chatbot.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
load_dotenv()

VECTOR_DB_PATH  = "ecommerce_faiss_index"
TOP_K_RETRIEVAL = 5          # number of chunks to retrieve per query
MAX_HISTORY     = 10         # max turns of conversation to keep in context
MODEL_NAME      = "gpt-4o-mini"

# ─────────────────────────────────────────────
# Load FAISS Vector Store
# ─────────────────────────────────────────────
if not os.path.exists(VECTOR_DB_PATH):
    raise RuntimeError(
        f"Vector store not found at '{VECTOR_DB_PATH}/'.\n"
        "Please run  python ingest.py  first to build the knowledge base."
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_API_BASE"),
)

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_RETRIEVAL},
)

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable and friendly E-Commerce Support Assistant for ShopEasy.

Your responsibilities:
- Help customers with product questions, return/refund queries, shipping enquiries, warranty claims,
  account issues, and any topic covered in the provided documents.
- Answer ONLY using the information found in the retrieved document context below.
- If the answer is not found in the provided documents, respond with exactly:
  "I don't have enough information in the provided documents."
- Do NOT make up policies, dates, phone numbers, or procedures not present in the documents.
- Be concise, warm, and professional.
- When referencing policies, quote the relevant section briefly so the customer knows exactly
  what policy applies.
- Use bullet points for multi-step processes or lists.

Conversation history is provided so you can answer follow-up questions coherently.
"""

HUMAN_TEMPLATE = """--- Retrieved Document Context ---
{context}
--- End of Context ---

Customer Question: {question}"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", HUMAN_TEMPLATE),
])

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_context(question: str) -> tuple[str, list]:
    """Retrieve relevant chunks and format them as a context string."""
    docs = retriever.invoke(question)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Document {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(context_parts), docs


def trim_history(history: list, max_turns: int) -> list:
    """Keep only the most recent `max_turns` (human + AI) message pairs."""
    # Each turn = 1 HumanMessage + 1 AIMessage = 2 messages
    cutoff = max_turns * 2
    return history[-cutoff:] if len(history) > cutoff else history


def get_response(question: str, history: list) -> str:
    """Run the RAG pipeline and return the assistant's reply."""
    context, _ = build_context(question)

    messages = prompt.format_messages(
        context=context,
        question=question,
        history=history,
    )

    response = llm.invoke(messages)
    return response.content


# ─────────────────────────────────────────────
# Chat Loop
# ─────────────────────────────────────────────
def chat():
    print("\n" + "=" * 65)
    print("  ShopEasy Customer Support Chatbot")
    print("  Powered by RAG + GPT-4o-mini")
    print("  Type 'exit' to quit | Type 'clear' to reset conversation")
    print("=" * 65 + "\n")

    conversation_history: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Have a great shopping experience. 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\nThank you for using ShopEasy Support. Goodbye! 👋\n")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("\n[Conversation history cleared.]\n")
            continue

        # Get response
        reply = get_response(user_input, conversation_history)

        print(f"\nShopEasy Bot: {reply}\n")
        print("-" * 65)

        # Update history
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(AIMessage(content=reply))
        conversation_history = trim_history(conversation_history, MAX_HISTORY)


if __name__ == "__main__":
    chat()
