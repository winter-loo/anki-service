import os
import importlib

from fastapi.testclient import TestClient


def test_collections_fs_list_create_delete(tmp_path, monkeypatch):
    # Configure env before importing web_api
    monkeypatch.setenv("ANKI_AUTH_MODE", "dev_header")
    # keep the setting local to this test file
    monkeypatch.setenv("ANKI_DATA_DIR", str(tmp_path))

    import web_api

    importlib.reload(web_api)

    client = TestClient(web_api.api_app)

    # empty
    r = client.get("/collections", headers={"X-User-Id": "u1"})
    assert r.status_code == 200
    assert r.json() == []

    # create default random id
    r = client.post("/collections", json={"name": "My Deck"}, headers={"X-User-Id": "u1"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "collection_id" in data
    cid = data["collection_id"]

    # list contains it
    r = client.get("/collections", headers={"X-User-Id": "u1"})
    assert r.status_code == 200
    items = r.json()
    assert any(it["collection_id"] == cid for it in items)

    # create explicit id
    r = client.post(
        "/collections",
        json={"collection_id": "c-1", "name": "C1"},
        headers={"X-User-Id": "u1"},
    )
    assert r.status_code == 200

    # users isolated
    r = client.get("/collections", headers={"X-User-Id": "u2"})
    assert r.status_code == 200
    assert r.json() == []

    # delete
    r = client.delete(f"/collections/{cid}", headers={"X-User-Id": "u1"})
    assert r.status_code == 200

    # deleted from disk
    assert not os.path.exists(tmp_path / "users" / "u1" / "collections" / cid)

    # delete again -> 404
    r = client.delete(f"/collections/{cid}", headers={"X-User-Id": "u1"})
    assert r.status_code == 404
