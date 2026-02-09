from __future__ import annotations

# NOTE: The Anki Rust bridge modules require a compiled extension.
# We keep auth-related endpoints importable even if the extension isn't built
# (useful for lightweight auth verification tests).
try:
    from anki._backend import RustBackend  # type: ignore
    import anki.search_pb2  # type: ignore
    import anki.cards_pb2  # type: ignore
    import anki.scheduler_pb2  # type: ignore
except Exception:  # pragma: no cover
    RustBackend = None  # type: ignore
    anki = None  # type: ignore

from fastapi import Depends, FastAPI, Header, HTTPException
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
#
# Clients call this API with:
#   Authorization: Bearer <supabase_access_token>
#
# Authentication is Supabase-only.
AUTH_MODE = "supabase"
# Storage base directory (attach a persistent volume here in production)
DATA_DIR = os.environ.get("ANKI_DATA_DIR", os.environ.get("ANKI_USER_BASE_DIR", "users_data"))
DEFAULT_COLLECTION_ID = os.environ.get("ANKI_DEFAULT_COLLECTION_ID", "default")

# Env vars:
# - PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
# - PUBLIC_SUPABASE_PUBLISHABLE_KEY=<publishable key>
#   - also supports SUPABASE_PUBLISHABLE_KEY or legacy SUPABASE_ANON_KEY
# - SUPABASE_SECRET_KEY (optional; server-side admin endpoints)
SUPABASE_PROJECT_URL = os.environ.get("PUBLIC_SUPABASE_URL")
SUPABASE_PUBLISHABLE_KEY = (
    os.environ.get("PUBLIC_SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
)

# Server-side admin key (keep secret). Used only for optional signup policy/metrics.
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
    # Keep it simple: allow UUIDs and common slugs.
    # (No slashes, no dots-only, no spaces)
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
        raise HTTPException(
            500,
            "Supabase auth enabled but PUBLIC_SUPABASE_URL is not configured.",
        )

    key = SUPABASE_PUBLISHABLE_KEY or SUPABASE_SECRET_KEY
    if not key:
        raise HTTPException(
            500,
            (
                "Supabase auth enabled but no key configured. "
                "Set PUBLIC_SUPABASE_PUBLISHABLE_KEY (or SUPABASE_PUBLISHABLE_KEY / SUPABASE_ANON_KEY)."
            ),
        )

    if _SUPABASE_CLIENT is None:
        create_client = _import_supabase_create_client()
        _SUPABASE_CLIENT = create_client(SUPABASE_PROJECT_URL, key)

    response = _SUPABASE_CLIENT.auth.get_claims(jwt=token)
    return _extract_supabase_claims(response)


def require_user_id(
    authorization: str | None = Header(default=None),
) -> str:
    """Resolve user id from an authenticated identity.

    Authorization Bearer token -> Supabase get_claims -> subject claim (sub).
    """
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
class UserBackend:
    backend: RustBackend
    lock: threading.Lock


class BackendManager:
    """Caches a RustBackend per (user_id, collection_id) and protects SQLite access with a lock."""

    def __init__(self) -> None:
        _ensure_dir(DATA_DIR)
        self._items: dict[tuple[str, str], UserBackend] = {}
        self._guard = threading.Lock()

    def evict(self, user_id: str, collection_id: str) -> None:
        """Remove a cached backend (best-effort close)."""
        key = (str(user_id), str(collection_id))
        tb: UserBackend | None = None
        with self._guard:
            tb = self._items.pop(key, None)
        if tb is not None:
            try:
                with tb.lock:
                    # close_collection expects a bool argument
                    tb.backend.close_collection(downgrade_to_schema11=False)
            except Exception:
                # Best-effort: don't crash deletion paths if closing fails.
                pass

    def _collection_paths(self, user_id: str, collection_id: str) -> tuple[str, str, str, str]:
        col_dir = _user_collection_dir(user_id, collection_id)
        _ensure_dir(col_dir)
        collection_path = os.path.join(col_dir, "collection.anki2")
        media_folder_path = os.path.join(col_dir, "collection.media")
        media_db_path = os.path.join(col_dir, "collection.media.db2")
        log_dir = os.path.join(col_dir, "log")
        _ensure_dir(media_folder_path)
        _ensure_dir(log_dir)
        return col_dir, collection_path, media_folder_path, media_db_path

    def get(self, user_id: str, collection_id: str) -> UserBackend:
        key = (str(user_id), str(collection_id))
        with self._guard:
            if key in self._items:
                return self._items[key]

            if RustBackend is None:
                raise RuntimeError(
                    "Anki RustBackend is not available (missing compiled extension)."
                )

            col_dir, collection_path, media_folder_path, media_db_path = self._collection_paths(user_id, collection_id)

            # Initialize backend per user.
            # Logging initialization is global-ish, but we can still point it at the
            # user's directory for easier debugging.
            # Passing a custom log directory path can fail on some platforms/builds.
            # Default logging setup is sufficient for the web API.
            RustBackend.initialize_logging(None)
            bk = RustBackend(["en"], True)
            bk.open_collection(
                collection_path=collection_path,
                media_folder_path=media_folder_path,
                media_db_path=media_db_path,
            )

            item = UserBackend(backend=bk, lock=threading.Lock())
            self._items[key] = item
            return item


backend_manager = BackendManager()


def require_collection_id(x_collection_id: str | None = Header(default=None)) -> str:
    """Resolve collection id.

    For multi-collection support we take it from a header. This avoids duplicating every API route
    under a path prefix.

    Default: "default".
    """
    return x_collection_id or DEFAULT_COLLECTION_ID


def get_bk(
    user_id: str = Depends(require_user_id),
    collection_id: str = Depends(require_collection_id),
) -> UserBackend:
    return backend_manager.get(user_id, collection_id)


# -----------------------------
# FastAPI apps
# -----------------------------
api_app = FastAPI(title="Anki Web API")


@api_app.get("/auth/whoami")
def auth_whoami(
    user_id: str = Depends(require_user_id),
    collection_id: str = Depends(require_collection_id),
) -> dict:
    """Lightweight auth check endpoint (no Anki backend required)."""
    return {
        "user_id": user_id,
        "collection_id": collection_id,
        "auth_mode": AUTH_MODE,
    }


class SignupRequest(BaseModel):
    email: str
    password: str


def _supabase_admin_request(path: str, *, method: str = "GET", json_body: dict | None = None) -> tuple[int, dict, dict]:
    """Minimal Supabase Auth admin REST helper.

    Returns (status_code, headers_lower, json_data).
    """
    project_url = os.environ.get("PUBLIC_SUPABASE_URL")
    if not project_url or not SUPABASE_SECRET_KEY:
        raise HTTPException(500, "Supabase admin is not configured (set PUBLIC_SUPABASE_URL and SUPABASE_SECRET_KEY)")

    import urllib.request
    import urllib.error

    url = project_url.rstrip("/") + path
    body = None
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")

    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("apikey", SUPABASE_SECRET_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_SECRET_KEY}")
    req.add_header("content-type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, headers, data
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if hasattr(e, "read") else ""
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {"error": raw}
        headers = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
        return int(getattr(e, "code", 500)), headers, data


def _supabase_user_count() -> int:
    """Return current number of users in Supabase Auth.

    We try to read total from Content-Range if present; otherwise we fall back to paging.
    """
    # Try: per_page=1 and parse Content-Range total, if provided.
    status, headers, data = _supabase_admin_request("/auth/v1/admin/users?page=1&per_page=1")
    if status >= 400:
        raise HTTPException(500, f"Supabase admin users list failed: {data}")

    cr = headers.get("content-range") or headers.get("content_range")
    if cr and "/" in cr:
        try:
            return int(cr.split("/", 1)[1])
        except Exception:
            pass

    # Fallback: page until empty (stop early once >1001)
    total = 0
    page = 1
    per_page = 200
    while True:
        status, _, items = _supabase_admin_request(f"/auth/v1/admin/users?page={page}&per_page={per_page}")
        if status >= 400:
            raise HTTPException(500, f"Supabase admin users list failed: {items}")
        if not isinstance(items, list) or len(items) == 0:
            break
        total += len(items)
        if total > 1100:
            break
        page += 1
    return total


@api_app.post("/public/signup")
def public_signup(req: SignupRequest) -> dict:
    """Controlled signup policy:

    - If user_count <= 1001: create user via admin API with email_confirm=true (no email confirmation needed).
    - If user_count > 1001: require email confirmation (client should use normal Supabase signUp).

    NOTE: For the >1001 case to work, enable "Confirm email" in Supabase Auth settings.
    """
    count = _supabase_user_count()
    if count > 1001:
        return {"mode": "require_email_confirm", "user_count": count, "threshold": 1001}

    status, _, data = _supabase_admin_request(
        "/auth/v1/admin/users",
        method="POST",
        json_body={"email": req.email, "password": req.password, "email_confirm": True},
    )
    if status >= 400:
        # pass through useful error
        raise HTTPException(400, f"admin create user failed: {data}")

    return {"mode": "created_confirmed", "user_count": count, "threshold": 1001, "user_id": data.get("id")}
app = FastAPI(title="main app")


@app.on_event("startup")
def _print_startup_env() -> None:
    print("[startup] environment variables:", flush=True)
    for key in sorted(os.environ):
        print(f"{key}={os.environ.get(key, '')}", flush=True)


# Set up CORS middleware
origins = [
    "*",
    # local unpack extension on Mac
    "chrome-extension://oglpjlknjdpkmcajnopbkafkdbieolpj",
    # local unpack extension on Windows 10
    "chrome-extension://ajaafalcmjeakneghkgaklhnhbaphaai",
    # chrome web store extension id
    "chrome-extension://gcgjhmoegecipjkgpkddhobjfjnagoco",
]

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_app.get("/collections")
def list_collections(user_id: str = Depends(require_user_id)) -> list[dict]:
    """List collections for the authenticated user (filesystem-backed)."""
    base = os.path.join(DATA_DIR, "users", _safe_id(user_id, what="user_id"), "collections")
    if not os.path.isdir(base):
        return []

    out: list[dict] = []
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        # validate dir name is a safe id
        try:
            cid = _safe_id(name, what="collection_id")
        except Exception:
            continue

        meta_path = os.path.join(full, "meta.json")
        meta = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None

        out.append({"collection_id": cid, "meta": meta})
    return out


@api_app.post("/collections")
def create_collection(
    req: CreateCollectionRequest,
    user_id: str = Depends(require_user_id),
) -> dict:
    """Create a new collection directory for the user.

    The collection DB is created lazily when the first Anki API call opens it.
    If you want to force-create/validate the collection immediately, use:
      POST /collections/{collection_id}/init
    """
    # Choose id
    if req.collection_id:
        cid = _safe_id(req.collection_id, what="collection_id")
    else:
        # short random id
        cid = base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")
        cid = _safe_id(cid, what="collection_id")

    col_dir = _user_collection_dir(user_id, cid)
    if os.path.exists(col_dir):
        raise HTTPException(409, "collection already exists")

    _ensure_dir(col_dir)
    _ensure_dir(os.path.join(col_dir, "collection.media"))

    meta = {
        "name": req.name or cid,
        "created_at": int(time.time()),
    }
    try:
        with open(os.path.join(col_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception:
        # meta is optional
        pass

    return {"collection_id": cid, "meta": meta}


@api_app.post("/collections/{collection_id}/init")
def init_collection(
    collection_id: str,
    user_id: str = Depends(require_user_id),
) -> dict:
    """Force-initialize a collection by opening it via RustBackend.

    Useful to validate the environment and ensure the underlying SQLite DB files exist.
    """
    cid = _safe_id(collection_id, what="collection_id")
    if RustBackend is None:
        raise HTTPException(500, "Anki RustBackend is not available. Run ./build_anki first.")

    # Open/initialize backend (this will create collection.anki2 if missing)
    tb = backend_manager.get(user_id, cid)
    # Close it again to avoid keeping too many open collections around
    backend_manager.evict(user_id, cid)

    col_dir = _user_collection_dir(user_id, cid)
    return {
        "initialized": True,
        "collection_id": cid,
        "path": col_dir,
    }


@api_app.delete("/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    user_id: str = Depends(require_user_id),
) -> dict:
    """Delete a collection directory.

    Best-effort closes any cached backend first.
    """
    cid = _safe_id(collection_id, what="collection_id")
    backend_manager.evict(user_id, cid)

    col_dir = _user_collection_dir(user_id, cid)
    if not os.path.isdir(col_dir):
        raise HTTPException(404, "collection not found")

    shutil.rmtree(col_dir)
    return {"deleted": True, "collection_id": cid}


class UserNote(BaseModel):
    fields: list[str]


class CreateCollectionRequest(BaseModel):
    collection_id: str | None = None
    name: str | None = None


@api_app.get("/note/list")
def list_notes(tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        deck = tb.backend.get_current_deck()
        sn = anki.search_pb2.SearchNode(deck=deck.name)
        ss = tb.backend.build_search_string(sn)
        so = anki.search_pb2.SortOrder(
            builtin=anki.search_pb2.SortOrder.Builtin(column="noteCrt")
        )
        note_id_list = tb.backend.search_notes(search=ss, order=so)
        resp = []
        for nid in note_id_list:
            note = tb.backend.get_note(nid)
            resp.append(MessageToDict(note))
        return resp


@api_app.post("/note/add/{fld}")
def create_note(fld: str, tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        basic_notetype = tb.backend.get_notetype_names()[0]
        nn = tb.backend.new_note(basic_notetype.id)
        nn.fields[0] = fld
        resp = tb.backend.add_note(note=nn, deck_id=tb.backend.get_current_deck().id)
        return {"note_id": resp.note_id}


@api_app.post("/note/add")
def create_note_by_json(new_user_note: UserNote, tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        basic_notetype = tb.backend.get_notetype_names()[0]
        nn = tb.backend.new_note(basic_notetype.id)
        # RustBackend.new_note() returns a Note object with two fields
        nn.fields[0] = new_user_note.fields[0]
        if len(new_user_note.fields) > 1:
            nn.fields[1] = new_user_note.fields[1]
        for fld in new_user_note.fields[2:]:
            nn.fields.append(fld)
        resp = tb.backend.add_note(note=nn, deck_id=tb.backend.get_current_deck().id)
        return {"note_id": resp.note_id}


@api_app.post("/note/update/@{note_id}")
def update_note_by_id(note_id: int, user_note: UserNote, tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        note = tb.backend.get_note(note_id)
        note.fields[0] = user_note.fields[0]
        note.fields[1] = user_note.fields[1]
        resp = tb.backend.update_notes(notes=[note], skip_undo_entry=True)
        return MessageToDict(resp)


@api_app.get("/note/@{note_id}")
def read_note_by_id(note_id: int, tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        note = tb.backend.get_note(note_id)
        return MessageToDict(note)


@api_app.post("/note/delete/@{note_id}")
def delete_note_by_id(note_id: int, tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        card_ids = tb.backend.cards_of_note(nid=note_id)
        resp = tb.backend.remove_notes(note_ids=[note_id], card_ids=card_ids)
        return MessageToDict(resp)


@api_app.get("/note/studied_today")
def list_notes_studied_today(tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        resp = tb.backend.studied_today()
        return {"msg": resp}


@api_app.get("/card/sched_timing_today")
def get_scheduled_timing_today(tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        resp = tb.backend.sched_timing_today()
        return MessageToDict(resp)


@api_app.get("/card/next")
def get_next_card(tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        qcards = tb.backend.get_queued_cards(fetch_limit=1, intraday_learning_only=False)
        return MessageToDict(qcards)


def int_time(scale: int = 1) -> int:
    "The time in integer seconds. Pass scale=1000 to get milliseconds."
    return int(time.time() * scale)


def build_answer(
    *,
    card: anki.cards_pb2.Card,
    states: anki.scheduler_pb2.SchedulingStates,
    rating: anki.scheduler_pb2.CardAnswer.Rating
) -> anki.scheduler_pb2.CardAnswer:
    "Build input for answer_card()."
    if rating == anki.scheduler_pb2.CardAnswer.AGAIN:
        new_state = states.again
    elif rating == anki.scheduler_pb2.CardAnswer.HARD:
        new_state = states.hard
    elif rating == anki.scheduler_pb2.CardAnswer.GOOD:
        new_state = states.good
    elif rating == anki.scheduler_pb2.CardAnswer.EASY:
        new_state = states.easy
    else:
        raise Exception("invalid rating")

    return anki.scheduler_pb2.CardAnswer(
        card_id=card.id,
        current_state=states.current,
        new_state=new_state,
        rating=rating,
        answered_at_millis=int_time(1000),
        milliseconds_taken=0,
    )


def rating_from_ease(ease):
    if ease == 1:
        return anki.scheduler_pb2.CardAnswer.AGAIN
    elif ease == 2:
        return anki.scheduler_pb2.CardAnswer.HARD
    elif ease == 3:
        return anki.scheduler_pb2.CardAnswer.GOOD
    else:
        return anki.scheduler_pb2.CardAnswer.EASY


@api_app.post("/card/answer/{ease}")
def answer_card(ease: int, tb: UserBackend = Depends(get_bk)):
    with tb.lock:
        qcards = tb.backend.get_queued_cards(fetch_limit=1, intraday_learning_only=False)
        if len(qcards.cards) == 0:
            return {}
        top_card = qcards.cards[0].card
        current_states = qcards.cards[0].states
        answer = build_answer(
            card=top_card,
            states=current_states,
            rating=rating_from_ease(ease),
        )

        resp = tb.backend.answer_card(answer)
        return MessageToDict(resp)


@app.get("/explain/{text}")
def explain_word(text: str, model: str = "gemini-2.5-flash-lite"):
    if get_word_explanation is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Word explanation service is unavailable. "
                f"Reason: {_word_explanation_import_error}."
            ),
        )
    result = get_word_explanation(text, model=model)
    return result


@app.get("/ui-config.json")
def ui_config() -> dict:
    """Expose safe public config needed by the static UI.

    Keep this endpoint free of secrets.
    """
    return {
        "supabaseUrl": os.environ.get("PUBLIC_SUPABASE_URL") or "",
        "supabasePublishableKey": os.environ.get("PUBLIC_SUPABASE_PUBLISHABLE_KEY") or "",
    }


app.mount("/api", api_app)

# Static web UI
# - Source: ui/web (SvelteKit)
# - Build output: ui/out (not committed)
UI_DIR = os.environ.get("ANKI_UI_DIR", "ui/out")
os.makedirs(UI_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
