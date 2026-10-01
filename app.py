"""app.py - Streamlit chat UI. Run on Streamlit Cloud with: app.py as the main file."""
import os
import sys
import uuid

# Switch off CrewAI telemetry (it tries to phone home, which is noisy on Streamlit Cloud)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# Streamlit Cloud ships an old sqlite3 that some CrewAI dependencies (chromadb) reject.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

import escalation
from agent import run_agent

st.set_page_config(page_title="Customer Support", page_icon="🛍️", layout="centered")

# ---- secrets (Streamlit Cloud -> App settings -> Secrets) -------------------
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
MODEL = st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it in the app's Streamlit Secrets.")
    st.stop()

GREETING = "Assalam-o-Alaikum! 👋 I'm Sana from support. I can answer questions about our policies and track your orders. How can I help?"


def new_chat():
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]


if "session_id" not in st.session_state:
    new_chat()

sid = st.session_state.session_id

# ---- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("🛍️ Support")
    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.subheader("📋 Pending escalations")
    mine = escalation.list_tickets(session_id=sid)
    if not mine:
        st.caption("None for this chat.")
    for t in mine:
        with st.expander(f"{t['ticket_id']} - {t['status'].title()}"):
            st.write(f"**Order:** {t['order_id']}")
            st.write(f"**Customer:** {t['customer_name']}")
            st.write(f"**Reason:** {t['reason'].replace('_', ' ')}")
            st.write(f"**Summary:** {t['summary']}")
            st.caption(t["created_at"])

    # Optional staff view: only shown if ADMIN_PASSWORD is set in secrets
    if ADMIN_PASSWORD:
        st.divider()
        with st.expander("🔒 Staff dashboard"):
            if st.text_input("Password", type="password") == ADMIN_PASSWORD:
                all_pending = escalation.list_tickets()
                st.write(f"{len(all_pending)} pending ticket(s)")
                for t in all_pending:
                    st.markdown(f"**{t['ticket_id']}** · {t['customer_name']} · {t['order_id']}")
                    st.caption(t["summary"])
                    if st.button("Mark resolved", key=f"res_{t['ticket_id']}"):
                        escalation.resolve_ticket(t["ticket_id"])
                        st.rerun()

# ---- chat -------------------------------------------------------------------
st.title("Customer Support")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    history = list(st.session_state.messages)  # everything BEFORE this message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tickets_before = escalation.count_tickets(sid)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = run_agent(prompt, history, sid, API_KEY, MODEL)
            except Exception as exc:
                reply = "Sorry, something went wrong on my side. Please try again in a moment."
                st.caption(f"Technical detail: {type(exc).__name__}")
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    if escalation.count_tickets(sid) > tickets_before:
        st.rerun()  # refresh so the new ticket appears in the sidebar
