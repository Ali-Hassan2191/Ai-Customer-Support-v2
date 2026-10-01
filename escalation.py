"""
escalation.py - stores "escalate to a human" tickets.

Tickets are saved in a small JSON file in the system temp folder (so they are
never committed to GitHub). On Streamlit Cloud this file lives until the app
restarts - perfect for a demo. For production, swap these functions for a real
database / helpdesk API (Zendesk, Freshdesk, Google Sheets, ...).
"""
import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Asia/Karachi")
except Exception:  # tz database missing -> fall back to UTC
    _TZ = timezone.utc

TICKETS_FILE = Path(tempfile.gettempdir()) / "support_escalations.json"
_LOCK = threading.Lock()  # Streamlit can run several users at once


def _load() -> list:
    if not TICKETS_FILE.exists():
        return []
    try:
        return json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(tickets: list) -> None:
    TICKETS_FILE.write_text(
        json.dumps(tickets, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_ticket(
    session_id: str,
    summary: str,
    reason: str = "agent_unable",
    customer_name: str = "",
    order_id: str = "",
    conversation: list | None = None,
) -> dict:
    """Create a new pending ticket and return it."""
    with _LOCK:
        tickets = _load()
        ticket = {
            "ticket_id": f"TKT-{len(tickets) + 1:04d}",
            "session_id": session_id,
            "status": "pending",
            "reason": reason,  # "user_requested" or "agent_unable"
            "customer_name": customer_name or "Unknown",
            "order_id": order_id or "-",
            "summary": summary,
            "recent_conversation": (conversation or [])[-6:],
            "created_at": datetime.now(_TZ).strftime("%d-%m-%Y %H:%M"),
        }
        tickets.append(ticket)
        _save(tickets)
    return ticket


def list_tickets(session_id: str | None = None, status: str = "pending") -> list:
    """Tickets with the given status. Pass session_id to see only one chat's tickets."""
    with _LOCK:
        tickets = _load()
    return [
        t
        for t in tickets
        if t["status"] == status and (session_id is None or t["session_id"] == session_id)
    ]


def count_tickets(session_id: str) -> int:
    with _LOCK:
        return sum(1 for t in _load() if t["session_id"] == session_id)


def resolve_ticket(ticket_id: str) -> None:
    with _LOCK:
        tickets = _load()
        for t in tickets:
            if t["ticket_id"] == ticket_id:
                t["status"] = "resolved"
        _save(tickets)
