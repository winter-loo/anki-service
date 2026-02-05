import os
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _JWKSHandler(BaseHTTPRequestHandler):
    jwks = b"{}"

    def do_GET(self):
        if self.path.endswith("/.well-known/jwks.json"):
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(self.jwks)
            return
        self.send_response(404)
        self.end_headers()


def _make_rsa_jwk_and_token(sub: str, issuer: str, jwks_url: str):
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()

    def b64(n: int) -> str:
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        import base64

        return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

    jwk = {
        "kty": "RSA",
        "kid": "k1",
        "use": "sig",
        "alg": "RS256",
        "n": b64(public_numbers.n),
        "e": b64(public_numbers.e),
    }

    token = jwt.encode(
        {
            "sub": sub,
            "iss": issuer,
            "aud": "authenticated",
            "iat": 1,
            "exp": 9999999999,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    return jwk, token


@pytest.mark.timeout(30)
def test_supabase_auth_mode_whoami(tmp_path):
    # Start JWKS HTTP server
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    issuer = f"http://127.0.0.1:{port}/auth/v1"

    jwk, token = _make_rsa_jwk_and_token(
        sub="user-123",
        issuer=issuer,
        jwks_url=f"{issuer}/.well-known/jwks.json",
    )

    _JWKSHandler.jwks = json.dumps({"keys": [jwk]}).encode("utf-8")
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Configure env before import
    os.environ["ANKI_AUTH_MODE"] = "supabase"
    os.environ["PUBLIC_SUPABASE_URL"] = f"http://127.0.0.1:{port}"  # issuer derived as /auth/v1
    os.environ.pop("SUPABASE_JWT_ISSUER", None)
    os.environ.pop("SUPABASE_JWKS_URL", None)
    os.environ["ANKI_DATA_DIR"] = str(tmp_path)

    import sys, importlib

    sys.path.insert(0, os.path.abspath("pylib"))
    sys.path.insert(0, os.path.abspath("."))

    web_api = importlib.import_module("web_api")
    web_api = importlib.reload(web_api)

    from fastapi.testclient import TestClient

    client = TestClient(web_api.api_app)
    r = client.get("/auth/whoami", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["user_id"] == "user-123"
    assert data["auth_mode"] == "supabase"

    server.shutdown()
