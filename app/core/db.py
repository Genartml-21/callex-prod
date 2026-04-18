import asyncio
from typing import Dict, Any, Optional, List
from firebase_admin import firestore as fs

# ───────── Asynchronous Firestore Wrappers ─────────
# These functions wrap Google's synchronous Firestore SDK
# in thread pools to prevent blocking the WebSocket audio async loops.

def _get_db():
    from app.core.agent_loader import _get_db as centralized_db
    return centralized_db()

async def db_get_doc(collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a document entirely asynchronously."""
    def _fetch():
        db = _get_db()
        doc = db.collection(collection).document(str(doc_id)).get()
        return doc.to_dict() if doc.exists else None
    return await asyncio.to_thread(_fetch)

async def db_set_doc(collection: str, doc_id: str, data: Dict[str, Any]):
    """Set a document asynchronously without blocking."""
    def _set():
        db = _get_db()
        db.collection(collection).document(str(doc_id)).set(data)
    await asyncio.to_thread(_set)

async def db_update_doc(collection: str, doc_id: str, data: Dict[str, Any]):
    """Update an existing document natively in the background."""
    def _update():
        db = _get_db()
        doc_ref = db.collection(collection).document(str(doc_id))
        if doc_ref.get().exists:
            doc_ref.update(data)
    await asyncio.to_thread(_update)

async def db_add_doc(collection: str, data: Dict[str, Any]) -> str:
    """Add a document and return its auto-generated ID without blocking."""
    def _add():
        db = _get_db()
        ref = db.collection(collection).document()
        ref.set(data)
        return ref.id
    return await asyncio.to_thread(_add)

async def db_query_where(collection: str, field: str, op: str, value: Any) -> List[Dict[str, Any]]:
    """Query a collection securely out-of-thread."""
    def _query():
        db = _get_db()
        snap = db.collection(collection).where(field, op, value).get()
        return [doc.to_dict() for doc in snap]
    return await asyncio.to_thread(_query)
