import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _make_rsa_jwk_and_token(sub: str, issuer: str, audience: str | None, jwks_url: str):
    import jwt
    from jwt.algorithms import RSAAlgorithm
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    kid = "test-key-1"
    jwk_json = RSAAlgorithm.to_jwk(public_key)
    # jwk_json is a JSON string
    import json

    jwk = json.loads(jwk_json)
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})

    payload = {"sub": sub, "iss": issuer}
    if audience:
        payload["aud"] = audience

    token = jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})
    return jwk, token


class _JWKSHandler(BaseHTTPRequestHandler):
    jwks = None

    def do_GET(self):
        if self.path != "/.well-known/jwks.json":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.jwks)

    def log_message(self, format, *args):
        return


@pytest.mark.timeout(30)
def test_rs256_jwks_auth_whoami():
    # Start JWKS HTTP server
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    issuer = f"http://127.0.0.1:{port}"

    jwk, token = _make_rsa_jwk_and_token(
        sub="tenant-123",
        issuer=issuer,
        audience=None,
        jwks_url=f"{issuer}/.well-known/jwks.json",
    )

    import json

    _JWKSHandler.jwks = json.dumps({"keys": [jwk]}).encode("utf-8")

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Configure app env before import
    os.environ["ANKI_AUTH_MODE"] = "jwt"
    os.environ["ANKI_JWT_ALG"] = "RS256"
    os.environ["ANKI_JWT_ISSUER"] = issuer
    os.environ.pop("ANKI_JWKS_URL", None)  # should derive from issuer

    # Import after env is set
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
    assert data["user_id"] == "tenant-123"
    assert data["jwt_alg"] == "RS256"

    server.shutdown()
