from __future__ import annotations

# NOTE: The Anki Rust bridge modules require a compiled extension.
# We keep auth-related endpoints importable even if the extension isn't built
# (useful for lightweight tests like RS256/JWKS auth verification).
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
except Exception:  # pragma: no cover
    get_word_explanation = None  # type: ignore

import base64
import json
import os
import threading
import time
from dataclasses import dataclass


# -----------------------------
# Multi-tenancy + Auth (best practice)
# -----------------------------
#
# The frontend (e.g. Better Auth) should authenticate the user and call this API with
#   Authorization: Bearer <token>
#
# This service then maps the authenticated user/org to a *tenant*, and uses a separate
# Anki collection per tenant:
#   tenants/<tenant_id>/collection.anki2
#   tenants/<tenant_id>/collection.media/
#
# NOTE: Proper JWT signature verification depends on your auth provider (issuer/JWKS/etc).
# This implementation supports:
# - Production: set ANKI_AUTH_MODE=jwt and plug in your verifier later
# - Development: set ANKI_AUTH_MODE=dev_header and send X-User-Id
#
# In all cases, we never trust an arbitrary tenant header unless explicitly enabled.

AUTH_MODE = os.environ.get("ANKI_AUTH_MODE", "dev_header")
TENANT_BASE_DIR = os.environ.get("ANKI_TENANT_BASE_DIR", "tenants")

# JWT verification settings.
#
# Default: RS256 + JWKS (recommended for production)
#
# Env vars:
# - ANKI_JWT_ALG: RS256 (default) or HS256
# - ANKI_JWKS_URL: JWKS endpoint URL (if RS256). If not set, we derive it from issuer:
#     <issuer>/.well-known/jwks.json
# - ANKI_JWT_ISSUER (or BETTER_AUTH_URL)
# - ANKI_JWT_AUDIENCE (optional)
# - ANKI_JWT_HS256_SECRET (or BETTER_AUTH_SECRET) if HS256
JWT_ALG = (os.environ.get("ANKI_JWT_ALG") or "RS256").upper()
JWT_HS256_SECRET = os.environ.get("ANKI_JWT_HS256_SECRET") or os.environ.get("BETTER_AUTH_SECRET")
JWT_ISSUER = os.environ.get("ANKI_JWT_ISSUER") or os.environ.get("BETTER_AUTH_URL")
JWT_AUDIENCE = os.environ.get("ANKI_JWT_AUDIENCE")
JWT_JWKS_URL = os.environ.get("ANKI_JWKS_URL")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _urlsafe_b64decode(data: str) -> bytes:
    data += "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def _jwt_unverified_claims(token: str) -> dict:
    """Extract claims without verifying signature."""
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("not a JWT")
    payload = json.loads(_urlsafe_b64decode(parts[1]))
    if not isinstance(payload, dict):
        raise ValueError("invalid JWT payload")
    return payload


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def verify_hs256_jwt(token: str, *, secret: str, issuer: str | None, audience: str | None) -> dict:
    """Verify an HS256 JWT locally (no external libs).

    Kept for local/dev or legacy setups.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT")
    header_b64, payload_b64, sig_b64 = parts

    header = json.loads(_urlsafe_b64decode(header_b64))
    if not isinstance(header, dict):
        raise ValueError("invalid header")
    if header.get("alg") != "HS256":
        raise ValueError(f"unsupported alg {header.get('alg')!r} (expected HS256)")

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    mac = __import__("hmac").new(secret.encode("utf-8"), signing_input, __import__("hashlib").sha256).digest()
    expected_sig = _b64url(mac)

    if not __import__("hmac").compare_digest(expected_sig, sig_b64):
        raise ValueError("bad signature")

    claims = json.loads(_urlsafe_b64decode(payload_b64))
    if not isinstance(claims, dict):
        raise ValueError("invalid payload")

    now = int(time.time())
    # Standard claim checks
    exp = claims.get("exp")
    if exp is not None and int(exp) < now:
        raise ValueError("token expired")
    nbf = claims.get("nbf")
    if nbf is not None and int(nbf) > now:
        raise ValueError("token not yet valid")

    if issuer and claims.get("iss") not in (issuer,):
        raise ValueError("issuer mismatch")

    if audience:
        aud = claims.get("aud")
        if isinstance(aud, str):
            ok = aud == audience
        elif isinstance(aud, list):
            ok = audience in aud
        else:
            ok = False
        if not ok:
            raise ValueError("audience mismatch")

    return claims


def _default_jwks_url(issuer: str | None) -> str | None:
    if not issuer:
        return None
    return issuer.rstrip("/") + "/.well-known/jwks.json"


def verify_rs256_jwt(token: str, *, jwks_url: str, issuer: str | None, audience: str | None) -> dict:
    """Verify an RS256 JWT via JWKS.

    Uses PyJWT's PyJWKClient to fetch + cache signing keys.
    """
    try:
        import jwt
        from jwt import PyJWKClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing PyJWT dependency: {e}")

    jwks_client = PyJWKClient(jwks_url)
    signing_key = jwks_client.get_signing_key_from_jwt(token).key

    # PyJWT will validate exp/nbf automatically unless options override.
    return jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        issuer=issuer,
        audience=audience,
        options={"verify_aud": bool(audience), "verify_iss": bool(issuer)},
    )


def require_tenant_id(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> str:
    """Resolve tenant id from an authenticated identity.

    Best practice: Authorization Bearer token -> verified -> subject claim (sub).

    Dev mode supported: X-User-Id header.
    """
    if AUTH_MODE == "dev_header":
        if not x_user_id:
            raise HTTPException(401, "Missing X-User-Id (dev mode)")
        return x_user_id

    if AUTH_MODE == "jwt":
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Missing Bearer token")
        token = authorization.split(" ", 1)[1].strip()

        try:
            if JWT_ALG == "HS256":
                if not JWT_HS256_SECRET:
                    raise HTTPException(
                        500,
                        "JWT auth enabled (HS256) but no secret configured. Set ANKI_JWT_HS256_SECRET (or BETTER_AUTH_SECRET).",
                    )
                claims = verify_hs256_jwt(
                    token,
                    secret=JWT_HS256_SECRET,
                    issuer=JWT_ISSUER,
                    audience=JWT_AUDIENCE,
                )
            elif JWT_ALG == "RS256":
                jwks_url = JWT_JWKS_URL or _default_jwks_url(JWT_ISSUER)
                if not jwks_url:
                    raise HTTPException(
                        500,
                        "JWT auth enabled (RS256) but no JWKS URL configured. Set ANKI_JWKS_URL (or ANKI_JWT_ISSUER/BETTER_AUTH_URL to derive /.well-known/jwks.json).",
                    )
                claims = verify_rs256_jwt(
                    token,
                    jwks_url=jwks_url,
                    issuer=JWT_ISSUER,
                    audience=JWT_AUDIENCE,
                )
            else:
                raise HTTPException(500, f"Unsupported ANKI_JWT_ALG={JWT_ALG!r}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(401, f"Invalid token: {e}")

        tenant_id = claims.get("sub")
        if not tenant_id:
            raise HTTPException(401, "Token missing 'sub'")
        return str(tenant_id)

    raise HTTPException(500, f"Unknown ANKI_AUTH_MODE={AUTH_MODE!r}")


@dataclass
class TenantBackend:
    backend: RustBackend
    lock: threading.Lock


class BackendManager:
    """Caches a RustBackend per tenant and protects SQLite access with a lock."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        _ensure_dir(self.base_dir)
        self._items: dict[str, TenantBackend] = {}
        self._guard = threading.Lock()

    def _tenant_paths(self, tenant_id: str) -> tuple[str, str, str, str]:
        tenant_dir = os.path.abspath(os.path.join(self.base_dir, tenant_id))
        _ensure_dir(tenant_dir)
        collection_path = os.path.join(tenant_dir, "collection.anki2")
        media_folder_path = os.path.join(tenant_dir, "collection.media")
        media_db_path = os.path.join(tenant_dir, "collection.media.db2")
        log_dir = os.path.join(tenant_dir, "log")
        _ensure_dir(media_folder_path)
        _ensure_dir(log_dir)
        return tenant_dir, collection_path, media_folder_path, media_db_path

    def get(self, tenant_id: str) -> TenantBackend:
        with self._guard:
            if tenant_id in self._items:
                return self._items[tenant_id]

            if RustBackend is None:
                raise RuntimeError(
                    "Anki RustBackend is not available (missing compiled extension). Run ./build_anki first."
                )

            _, collection_path, media_folder_path, media_db_path = self._tenant_paths(tenant_id)

            # Initialize backend per tenant.
            # Logging initialization is global-ish, but we can still point it at the
            # tenant's directory for easier debugging.
            RustBackend.initialize_logging(os.path.join(self.base_dir, tenant_id, "log"))
            bk = RustBackend(["en"], True)
            bk.open_collection(
                collection_path=collection_path,
                media_folder_path=media_folder_path,
                media_db_path=media_db_path,
                force_schema11=False,
            )

            item = TenantBackend(backend=bk, lock=threading.Lock())
            self._items[tenant_id] = item
            return item


backend_manager = BackendManager(TENANT_BASE_DIR)


def get_bk(tenant_id: str = Depends(require_tenant_id)) -> TenantBackend:
    return backend_manager.get(tenant_id)


# -----------------------------
# FastAPI apps
# -----------------------------
api_app = FastAPI(title="Anki Web API")


@api_app.get("/auth/whoami")
def auth_whoami(tenant_id: str = Depends(require_tenant_id)) -> dict:
    """Lightweight auth/tenant check endpoint (no Anki backend required)."""
    return {"tenant_id": tenant_id, "auth_mode": AUTH_MODE, "jwt_alg": JWT_ALG}
app = FastAPI(title="main app")

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


class UserNote(BaseModel):
    fields: list[str]


@api_app.get("/note/list")
def list_notes(tb: TenantBackend = Depends(get_bk)):
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
def create_note(fld: str, tb: TenantBackend = Depends(get_bk)):
    with tb.lock:
        basic_notetype = tb.backend.get_notetype_names()[0]
        nn = tb.backend.new_note(basic_notetype.id)
        nn.fields[0] = fld
        resp = tb.backend.add_note(note=nn, deck_id=tb.backend.get_current_deck().id)
        return {"note_id": resp.note_id}


@api_app.post("/note/add")
def create_note_by_json(new_user_note: UserNote, tb: TenantBackend = Depends(get_bk)):
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
def update_note_by_id(note_id: int, user_note: UserNote, tb: TenantBackend = Depends(get_bk)):
    with tb.lock:
        note = tb.backend.get_note(note_id)
        note.fields[0] = user_note.fields[0]
        note.fields[1] = user_note.fields[1]
        resp = tb.backend.update_notes(notes=[note], skip_undo_entry=True)
        return MessageToDict(resp)


@api_app.get("/note/@{note_id}")
def read_note_by_id(note_id: int, tb: TenantBackend = Depends(get_bk)):
    with tb.lock:
        note = tb.backend.get_note(note_id)
        return MessageToDict(note)


@api_app.post("/note/delete/@{note_id}")
def delete_note_by_id(note_id: int, tb: TenantBackend = Depends(get_bk)):
    with tb.lock:
        card_ids = tb.backend.cards_of_note(nid=note_id)
        resp = tb.backend.remove_notes(note_ids=[note_id], card_ids=card_ids)
        return MessageToDict(resp)


@api_app.get("/note/studied_today")
def list_notes_studied_today(tb: TenantBackend = Depends(get_bk)):
    with tb.lock:
        resp = tb.backend.studied_today()
        return {"msg": resp}


@api_app.get("/card/sched_timing_today")
def get_scheduled_timing_today(tb: TenantBackend = Depends(get_bk)):
    with tb.lock:
        resp = tb.backend.sched_timing_today()
        return MessageToDict(resp)


@api_app.get("/card/next")
def get_next_card(tb: TenantBackend = Depends(get_bk)):
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
def answer_card(ease: int, tb: TenantBackend = Depends(get_bk)):
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
    result = get_word_explanation(text, model=model)
    return result


app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="ui/web", html=True), name="ui")
