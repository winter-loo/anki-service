from __future__ import annotations

# NOTE: The Anki Rust bridge modules require a compiled extension.
# We keep auth-related endpoints importable even if the extension isn't built
# (useful for lightweight auth verification tests).
try:
    from anki.collection import Collection
    from anki.notes import NoteId
    from anki.cards import CardId
    from anki.decks import DeckId
    import anki.search_pb2  # type: ignore
    import anki.cards_pb2  # type: ignore
    import anki.scheduler_pb2  # type: ignore
except Exception:  # pragma: no cover
    Collection = None  # type: ignore
    NoteId = None  # type: ignore
    CardId = None  # type: ignore
    DeckId = None  # type: ignore
    anki = None  # type: ignore

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

try:
    from explain import get_word_explanation
    _word_explanation_import_error = None
except Exception as exc:  # pragma: no cover
    get_word_explanation = None  # type: ignore
    _word_explanation_import_error = str(exc)

import base64
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass


# -----------------------------
# Multi-user + Auth
# -----------------------------
AUTH_MODE = "supabase"
DATA_DIR = os.environ.get("ANKI_DATA_DIR", os.environ.get("ANKI_USER_BASE_DIR", "users_data"))
DEFAULT_COLLECTION_ID = os.environ.get("ANKI_DEFAULT_COLLECTION_ID", "default")

SUPABASE_PROJECT_URL = os.environ.get("PUBLIC_SUPABASE_URL")
SUPABASE_PUBLISHABLE_KEY = (
    os.environ.get("PUBLIC_SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
)
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")
_SUPABASE_CLIENT = None


class AuthDependencyError(RuntimeError):
    """Raised when optional auth dependencies are missing."""


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_id(value: str, *, what: str) -> str:
    """Validate a path component id (prevents traversal)."""
    if not value:
        raise ValueError(f"missing {what}")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    if any(ch not in allowed for ch in value):
        raise ValueError(f"invalid {what}: {value!r}")
    if "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def _user_collection_dir(user_id: str, collection_id: str) -> str:
    user_id = _safe_id(user_id, what="user_id")
    collection_id = _safe_id(collection_id, what="collection_id")
    return os.path.abspath(os.path.join(DATA_DIR, "users", user_id, "collections", collection_id))


def _import_supabase_create_client():
    try:
        from supabase import create_client
    except Exception as e:  # pragma: no cover
        raise AuthDependencyError(f"Missing supabase dependency: {e}") from e
    return create_client


def _mapping_from(value: object) -> dict | None:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return as_dict
    return None


def _extract_supabase_claims(response: object) -> dict:
    direct_claims = getattr(response, "claims", None)
    if isinstance(direct_claims, dict):
        return direct_claims
    direct_data = getattr(response, "data", None)
    if isinstance(direct_data, dict):
        nested_claims = direct_data.get("claims")
        if isinstance(nested_claims, dict):
            return nested_claims
        if "sub" in direct_data:
            return direct_data
    payload = _mapping_from(response) or {}
    nested_claims = payload.get("claims")
    if isinstance(nested_claims, dict):
        return nested_claims
    nested_data = payload.get("data")
    if isinstance(nested_data, dict):
        nested_claims = nested_data.get("claims")
        if isinstance(nested_claims, dict):
            return nested_claims
        if "sub" in nested_data:
            return nested_data
    if "sub" in payload:
        return payload
    raise ValueError("Supabase get_claims() did not return claims")


def _supabase_get_claims(token: str) -> dict:
    global _SUPABASE_CLIENT
    if not SUPABASE_PROJECT_URL:
        raise HTTPException(500, "Supabase auth enabled but PUBLIC_SUPABASE_URL is not configured.")
    key = SUPABASE_PUBLISHABLE_KEY or SUPABASE_SECRET_KEY
    if not key:
        raise HTTPException(500, "Supabase auth enabled but no key configured.")
    if _SUPABASE_CLIENT is None:
        create_client = _import_supabase_create_client()
        _SUPABASE_CLIENT = create_client(SUPABASE_PROJECT_URL, key)
    response = _SUPABASE_CLIENT.auth.get_claims(jwt=token)
    return _extract_supabase_claims(response)


def require_user_id(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = _supabase_get_claims(token)
    except HTTPException:
        raise
    except AuthDependencyError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(401, "Token missing 'sub'")
    return str(user_id)


@dataclass
class UserCollection:
    col: Collection
    lock: threading.Lock


class CollectionManager:
    def __init__(self) -> None:
        _ensure_dir(DATA_DIR)
        self._items: dict[tuple[str, str], UserCollection] = {}
        self._guard = threading.Lock()

    def evict(self, user_id: str, collection_id: str) -> None:
        key = (str(user_id), str(collection_id))
        uc: UserCollection | None = None
        with self._guard:
            uc = self._items.pop(key, None)
        if uc is not None:
            try:
                with uc.lock:
                    uc.col.close()
            except Exception:
                pass

    def _collection_paths(self, user_id: str, collection_id: str) -> tuple[str, str, str, str]:
        col_dir = _user_collection_dir(user_id, collection_id)
        _ensure_dir(col_dir)
        collection_path = os.path.join(col_dir, "collection.anki2")
        media_folder_path = os.path.join(col_dir, "collection.media")
        media_db_path = os.path.join(col_dir, "collection.media.db2")
        _ensure_dir(media_folder_path)
        return col_dir, collection_path, media_folder_path, media_db_path

    def get(self, user_id: str, collection_id: str) -> UserCollection:
        key = (str(user_id), str(collection_id))
        with self._guard:
            if key in self._items:
                return self._items[key]
            if Collection is None:
                raise RuntimeError("Anki Collection is not available.")
            _, collection_path, _, _ = self._collection_paths(user_id, collection_id)
            col = Collection(collection_path, server=True)
            item = UserCollection(col=col, lock=threading.Lock())
            self._items[key] = item
            return item


collection_manager = CollectionManager()


def require_collection_id(x_collection_id: str | None = Header(default=None)) -> str:
    return x_collection_id or DEFAULT_COLLECTION_ID


def get_col(user_id: str = Depends(require_user_id), collection_id: str = Depends(require_collection_id)) -> UserCollection:
    return collection_manager.get(user_id, collection_id)


# -----------------------------
# FastAPI apps
# -----------------------------
api_app = FastAPI(title="Anki Web API")

# --- Models ---

class UserNote(BaseModel):
    fields: list[str]
    deck_id: int | None = None

class CreateCollectionRequest(BaseModel):
    collection_id: str | None = None
    name: str | None = None

class CreateDeckRequest(BaseModel):
    name: str

class SignupRequest(BaseModel):
    email: str
    password: str

# --- API Endpoints (api_app) ---

@api_app.get("/auth/whoami")
def auth_whoami(user_id: str = Depends(require_user_id), collection_id: str = Depends(require_collection_id)):
    return {"user_id": user_id, "collection_id": collection_id, "auth_mode": AUTH_MODE}

@api_app.get("/collection/stats")
def get_collection_stats(uc: UserCollection = Depends(get_col)):
    with uc.lock:
        return {"note_count": uc.col.note_count(), "card_count": uc.col.card_count()}

@api_app.get("/collection")
def get_collection_info(uc: UserCollection = Depends(get_col)):
    with uc.lock:
        return {
            "path": uc.col.path,
            "note_count": uc.col.note_count(),
            "card_count": uc.col.card_count(),
            "current_deck": uc.col.decks.name(uc.col.decks.selected())
        }

@api_app.get("/collections")
def list_collections(user_id: str = Depends(require_user_id)):
    base = os.path.join(DATA_DIR, "users", _safe_id(user_id, what="user_id"), "collections")
    if not os.path.isdir(base): return []
    out = []
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if not os.path.isdir(full): continue
        try: cid = _safe_id(name, what="collection_id")
        except: continue
        meta_path = os.path.join(full, "meta.json")
        meta = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
            except: pass
        out.append({"collection_id": cid, "meta": meta})
    return out

@api_app.post("/collections")
def create_collection(req: CreateCollectionRequest, user_id: str = Depends(require_user_id)):
    cid = _safe_id(req.collection_id or base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("="), what="collection_id")
    col_dir = _user_collection_dir(user_id, cid)
    if os.path.exists(col_dir): raise HTTPException(409, "collection already exists")
    _ensure_dir(col_dir)
    _ensure_dir(os.path.join(col_dir, "collection.media"))
    meta = {"name": req.name or cid, "created_at": int(time.time())}
    try:
        with open(os.path.join(col_dir, "meta.json"), "w", encoding="utf-8") as f: json.dump(meta, f)
    except: pass
    return {"collection_id": cid, "meta": meta}

@api_app.post("/collections/{collection_id}/init")
def init_collection(collection_id: str, user_id: str = Depends(require_user_id)):
    cid = _safe_id(collection_id, what="collection_id")
    if Collection is None: raise HTTPException(500, "Anki Collection is not available.")
    uc = collection_manager.get(user_id, cid)
    collection_manager.evict(user_id, cid)
    return {"initialized": True, "collection_id": cid, "path": _user_collection_dir(user_id, cid)}

@api_app.delete("/collections/{collection_id}")
def delete_collection(collection_id: str, user_id: str = Depends(require_user_id)):
    cid = _safe_id(collection_id, what="collection_id")
    collection_manager.evict(user_id, cid)
    col_dir = _user_collection_dir(user_id, cid)
    if not os.path.isdir(col_dir): raise HTTPException(404, "collection not found")
    shutil.rmtree(col_dir)
    return {"deleted": True, "collection_id": cid}

@api_app.get("/deck/stats")
def get_deck_stats(deck_id: int | None = None, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        did = DeckId(deck_id) if deck_id is not None else uc.col.decks.selected()
        original_did = uc.col.decks.selected()
        try:
            if did != original_did: uc.col.decks.select(did)
            counts = uc.col.sched.counts()
            card_count = uc.col.db.scalar("select count() from cards where did=?", did)
            note_count = uc.col.db.scalar("select count() from notes where id in (select nid from cards where did=?)", did)
            return {
                "deck_id": did, "deck_name": uc.col.decks.name(did),
                "new_count": counts[0], "learning_count": counts[1], "review_count": counts[2],
                "note_count": note_count, "card_count": card_count,
            }
        finally:
            if did != original_did: uc.col.decks.select(original_did)

@api_app.get("/decks/current")
def get_current_deck(uc: UserCollection = Depends(get_col)):
    with uc.lock:
        did = uc.col.decks.selected()
        return {"id": did, "name": uc.col.decks.name(did)}

@api_app.get("/decks")
def list_decks(uc: UserCollection = Depends(get_col)):
    with uc.lock:
        return [MessageToDict(d) for d in uc.col.decks.all_names_and_ids()]

@api_app.post("/decks")
def create_deck(req: CreateDeckRequest, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        did = uc.col.decks.id(req.name)
        return {"id": did, "name": req.name}

@api_app.post("/decks/select/{deck_id}")
def select_deck(deck_id: int, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        uc.col.decks.select(DeckId(deck_id))
        return {"selected": True, "id": deck_id}

@api_app.delete("/decks/{deck_id}")
def delete_deck(deck_id: int, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        uc.col.decks.remove([DeckId(deck_id)])
        return {"deleted": True, "id": deck_id}

@api_app.get("/note/list")
def list_notes(deck_id: int | None = None, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        did = DeckId(deck_id) if deck_id is not None else uc.col.decks.selected()
        ss = uc.col.build_search_string(anki.search_pb2.SearchNode(deck=uc.col.decks.name(did)))
        note_id_list = uc.col.find_notes(ss, order="noteCrt asc")
        return [MessageToDict(uc.col.get_note(nid)._to_backend_note()) for nid in note_id_list]

@api_app.post("/note/add/{fld}")
def create_note(fld: str, deck_id: int | None = None, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        did = DeckId(deck_id) if deck_id is not None else uc.col.decks.selected()
        basic_notetype = uc.col.models.all_names_and_ids()[0]
        nn = uc.col.new_note(uc.col.models.get(basic_notetype.id))
        nn.fields[0] = fld
        uc.col.add_note(nn, did)
        return {"note_id": nn.id}

@api_app.post("/note/add")
def create_note_by_json(new_user_note: UserNote, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        did = DeckId(new_user_note.deck_id) if new_user_note.deck_id is not None else uc.col.decks.selected()
        basic_notetype = uc.col.models.all_names_and_ids()[0]
        nn = uc.col.new_note(uc.col.models.get(basic_notetype.id))
        nn.fields[0] = new_user_note.fields[0]
        if len(new_user_note.fields) > 1: nn.fields[1] = new_user_note.fields[1]
        for fld in new_user_note.fields[2:]: nn.fields.append(fld)
        uc.col.add_note(nn, did)
        return {"note_id": nn.id}

@api_app.post("/note/update/@{note_id}")
def update_note_by_id(note_id: int, user_note: UserNote, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        note = uc.col.get_note(NoteId(note_id))
        note.fields[0] = user_note.fields[0]
        note.fields[1] = user_note.fields[1]
        resp = uc.col.update_notes([note])
        return MessageToDict(resp)

@api_app.get("/note/@{note_id}")
def read_note_by_id(note_id: int, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        return MessageToDict(uc.col.get_note(NoteId(note_id))._to_backend_note())

@api_app.post("/note/delete/@{note_id}")
def delete_note_by_id(note_id: int, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        resp = uc.col.remove_notes([NoteId(note_id)])
        return MessageToDict(resp)

@api_app.get("/note/studied_today")
def list_notes_studied_today(uc: UserCollection = Depends(get_col)):
    with uc.lock: return {"msg": uc.col.studied_today()}

@api_app.get("/card/sched_timing_today")
def get_scheduled_timing_today(uc: UserCollection = Depends(get_col)):
    with uc.lock: return MessageToDict(uc.col.sched._timing_today())

@api_app.get("/card/next")
def get_next_card(deck_id: int | None = None, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        if deck_id is not None: uc.col.decks.select(DeckId(deck_id))
        return MessageToDict(uc.col.sched.get_queued_cards(fetch_limit=1, intraday_learning_only=False))

@api_app.get("/card/scheduling_states/@{card_id}")
def get_card_scheduling_states(card_id: int, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        states = uc.col.sched.get_scheduling_states(CardId(card_id))
        return {"states": MessageToDict(states), "labels": list(uc.col.sched.describe_next_states(states))}

@api_app.post("/card/answer/{ease}")
def answer_card(ease: int, uc: UserCollection = Depends(get_col)):
    with uc.lock:
        qcards = uc.col.sched.get_queued_cards(fetch_limit=1, intraday_learning_only=False)
        if not qcards.cards: return {}
        top_card, current_states = qcards.cards[0].card, qcards.cards[0].states
        answer = build_answer(card=top_card, states=current_states, rating=rating_from_ease(ease))
        return MessageToDict(uc.col.sched.answer_card(answer))

@api_app.post("/public/signup")
def public_signup(req: SignupRequest) -> dict:
    count = _supabase_user_count()
    if count > 1001: return {"mode": "require_email_confirm", "user_count": count, "threshold": 1001}
    status, _, data = _supabase_admin_request("/auth/v1/admin/users", method="POST",
        json_body={"email": req.email, "password": req.password, "email_confirm": True})
    if status >= 400: raise HTTPException(400, f"admin create user failed: {data}")
    return {"mode": "created_confirmed", "user_count": count, "threshold": 1001, "user_id": data.get("id")}

# --- Helpers ---

def int_time(scale: int = 1) -> int: return int(time.time() * scale)

def build_answer(*, card, states, rating) -> anki.scheduler_pb2.CardAnswer:
    if rating == anki.scheduler_pb2.CardAnswer.AGAIN: new_state = states.again
    elif rating == anki.scheduler_pb2.CardAnswer.HARD: new_state = states.hard
    elif rating == anki.scheduler_pb2.CardAnswer.GOOD: new_state = states.good
    elif rating == anki.scheduler_pb2.CardAnswer.EASY: new_state = states.easy
    else: raise Exception("invalid rating")
    return anki.scheduler_pb2.CardAnswer(card_id=card.id, current_state=states.current, new_state=new_state,
        rating=rating, answered_at_millis=int_time(1000), milliseconds_taken=0)

def rating_from_ease(ease):
    if ease == 1: return anki.scheduler_pb2.CardAnswer.AGAIN
    if ease == 2: return anki.scheduler_pb2.CardAnswer.HARD
    if ease == 3: return anki.scheduler_pb2.CardAnswer.GOOD
    return anki.scheduler_pb2.CardAnswer.EASY

# --- Main App ---

app = FastAPI(title="main app")

@app.on_event("startup")
def _print_startup_env() -> None:
    for key in sorted(os.environ): print(f"{key}={os.environ.get(key, '')}", flush=True)

origins = ["*", "chrome-extension://oglpjlknjdpkmcajnopbkafkdbieolpj", "chrome-extension://ajaafalcmjeakneghkgaklhnhbaphaai", "chrome-extension://gcgjhmoegecipjkgpkddhobjfjnagoco"]
api_app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/explain/{text}")
def explain_word(text: str, model: str = "gemini-2.5-flash-lite"):
    if not get_word_explanation: raise HTTPException(503, f"Service unavailable: {_word_explanation_import_error}")
    return get_word_explanation(text, model=model)

@app.get("/ui-config.json")
def ui_config() -> dict:
    return {"supabaseUrl": os.environ.get("PUBLIC_SUPABASE_URL") or "", "supabasePublishableKey": os.environ.get("PUBLIC_SUPABASE_PUBLISHABLE_KEY") or ""}

app.mount("/api", api_app)

UI_DIR = os.environ.get("ANKI_UI_DIR", "ui/out")
os.makedirs(UI_DIR, exist_ok=True)

@app.get("/api-lab", include_in_schema=False)
async def api_lab():
    return FileResponse(os.path.join(UI_DIR, "api.html"))

app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
