"""
tools.py - the 4 tools the agent can use.

1. search_knowledge_base  -> FAISS vector search over data/faiss.index + data/chunks.json
2. get_order_status       -> look up one order in data/orders.xlsx (needs identity check)
3. find_orders_by_phone   -> list a customer's orders from their phone number
4. escalate_to_human      -> create a pending ticket for a human agent
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from crewai.tools import tool

import escalation

DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # must match the model used to build faiss.index
TOP_K = 4

# ----------------------------------------------------------------------------
# 1) KNOWLEDGE BASE (FAISS)
# ----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_kb():
    """Loaded once, then reused (the embedding model is slow to load)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(DATA_DIR / "faiss.index"))
    raw = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))

    # chunks.json may be a plain list, or a dict like {"chunks": [...]}
    if isinstance(raw, dict):
        raw = raw.get("chunks", list(raw.values()))
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, raw, model


def _chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    for key in ("text", "content", "chunk_text", "page_content"):
        if key in chunk:
            return str(chunk[key])
    return ""


def _chunk_meta(chunk) -> dict:
    if not isinstance(chunk, dict):
        return {}
    meta = chunk.get("metadata")
    return meta if isinstance(meta, dict) else chunk  # metadata may be nested or flat


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base (policies, shipping, returns, warranty,
    payments, FAQs, product info). Use this for ANY question about how the
    company works. Input: a short natural-language search query."""
    try:
        index, chunks, model = _load_kb()
        vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(vec, TOP_K)
    except Exception as exc:  # missing file, model download problem, ...
        return f"KNOWLEDGE_BASE_ERROR: {exc}"

    results = []
    for rank, idx in enumerate(ids[0], start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        meta = _chunk_meta(chunk)
        source = meta.get("source_filename") or meta.get("source") or "unknown"
        page = meta.get("page_number") or meta.get("page") or "?"
        results.append(f"[Result {rank} | source: {source}, page: {page}]\n{_chunk_text(chunk)}")

    if not results:
        return "NO_RESULTS: nothing relevant found in the knowledge base."
    return "\n\n".join(results)


# ----------------------------------------------------------------------------
# 2) ORDERS (Excel used as a fake database)
# ----------------------------------------------------------------------------


def _normalize_order_id(raw: str) -> str:
    """'ord-2026-1001', '2026-1001' and '1001' all become 'ORD-2026-1001'."""
    s = re.sub(r"\s+", "", str(raw)).upper()
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2 and len(nums[-2]) == 4 and len(nums[-1]) == 4:
        return f"ORD-{nums[-2]}-{nums[-1]}"
    if nums and len(nums[-1]) == 4:
        return f"ORD-2026-{nums[-1]}"
    return s


def _normalize_phone(raw: str) -> str:
    """'+92 311 8143344', '0311-8143344', '3118143344' -> '03118143344'."""
    digits = re.sub(r"\D", "", str(raw))
    if digits.startswith("92") and len(digits) == 12:
        digits = "0" + digits[2:]
    elif len(digits) == 10 and digits.startswith("3"):
        digits = "0" + digits
    return digits


@lru_cache(maxsize=1)
def _orders() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "orders.xlsx")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
    df["Order Date"] = df["Order Date"].dt.strftime("%d-%m-%Y")
    for col in ["Order ID", "Customer Name", "Contact Number", "Product", "Shipping Address", "Status"]:
        df[col] = df[col].astype(str).str.strip()
    df["_oid"] = df["Order ID"].str.upper()
    df["_phone"] = df["Contact Number"].map(_normalize_phone)
    return df


def _identity_matches(row, proof: str) -> bool:
    """The customer must give the full name OR the phone number used on the order."""
    proof = (proof or "").strip()
    if not proof:
        return False
    if proof.lower() == row["Customer Name"].lower():
        return True
    phone = _normalize_phone(proof)
    return len(phone) >= 10 and phone == row["_phone"]


@tool("get_order_status")
def get_order_status(order_id: str, customer_name_or_phone: str) -> str:
    """Get the details and delivery status of ONE order. Both inputs are required:
    order_id (e.g. ORD-2026-1001) and customer_name_or_phone (the full name or the
    phone number used when ordering) to verify the customer's identity."""
    df = _orders()
    match = df[df["_oid"] == _normalize_order_id(order_id)]
    if match.empty:
        return "ORDER_NOT_FOUND: no order with that ID. Ask the customer to double-check it."

    row = match.iloc[0]
    if not _identity_matches(row, customer_name_or_phone):
        return (
            "VERIFICATION_FAILED: the name/phone does not match this order. Do NOT reveal any "
            "order details. Ask the customer for the full name or phone number used when ordering."
        )
    return (
        f"Order ID: {row['Order ID']}\n"
        f"Customer: {row['Customer Name']}\n"
        f"Product: {row['Product']}\n"
        f"Order Date: {row['Order Date']}\n"
        f"Status: {row['Status']}\n"
        f"Shipping Address: {row['Shipping Address']}"
    )


@tool("find_orders_by_phone")
def find_orders_by_phone(contact_number: str) -> str:
    """List all orders placed with a given phone number (e.g. 0311-8143344). Use this
    when the customer does not remember their order ID."""
    phone = _normalize_phone(contact_number)
    if len(phone) < 10:
        return "INVALID_PHONE: ask the customer for a valid Pakistani mobile number."
    df = _orders()
    rows = df[df["_phone"] == phone]
    if rows.empty:
        return "NO_ORDERS_FOUND for this phone number."
    lines = [
        f"- {r['Order ID']} | {r['Product']} | ordered {r['Order Date']} | status: {r['Status']}"
        for _, r in rows.iterrows()
    ]
    return "Orders found:\n" + "\n".join(lines)


# ----------------------------------------------------------------------------
# 3) ESCALATION (needs the chat session, so it is created by a small factory)
# ----------------------------------------------------------------------------


def get_tools(session_id: str, get_history=lambda: []):
    """Return the tool list for one chat session."""

    @tool("escalate_to_human")
    def escalate_to_human(
        summary: str,
        reason: str = "agent_unable",
        customer_name: str = "",
        order_id: str = "",
    ) -> str:
        """Hand the conversation to a human support agent. Use when the customer asks for
        a human, or when you cannot solve the problem with the other tools.
        summary: 2-4 sentence report of the customer's issue and what was already tried.
        reason: 'user_requested' if the customer asked for a human, else 'agent_unable'.
        customer_name / order_id: include them if known."""
        ticket = escalation.create_ticket(
            session_id=session_id,
            summary=summary,
            reason=reason,
            customer_name=customer_name,
            order_id=order_id,
            conversation=get_history(),
        )
        return (
            f"ESCALATED. Ticket {ticket['ticket_id']} created and marked pending. "
            "Tell the customer their request has been escalated to a human agent, "
            "give them the ticket ID, and say the team will contact them."
        )

    return [search_knowledge_base, get_order_status, find_orders_by_phone, escalate_to_human]
