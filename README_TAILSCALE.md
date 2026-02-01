# Tailscale dev access (Plan C)

Goal:
- Better Auth demo frontend (port 3000): reachable publicly via **Tailscale Funnel** for Google OAuth callbacks.
- anki-service (port 8000): reachable only inside the **tailnet** (your devices via Tailscale).

## URLs
- Better Auth base URL (public): `https://<node>.<tailnet>.ts.net/`
- anki-service (tailnet-only): `http://<tailscale-ip>:8000/`

## Prereqs
- `tailscaled` installed + logged in.
- Serve enabled in tailnet admin panel.
- If your network blocks Tailscale controlplane, configure a proxy for tailscaled (example below).

## Start

### Option 1: Better Auth demo (Google OAuth)

Set env vars:

```bash
export GOOGLE_CLIENT_ID='...'
export GOOGLE_CLIENT_SECRET='...'
```

Run:

```bash
cd /home/ldd/anki-service
./run_demo_tailscale.sh
```

### Option 2: Supabase Auth demo (recommended)

Set env vars:

```bash
export PUBLIC_SUPABASE_URL='https://<ref>.supabase.co'
export PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY='sb_publishable_...'
```

Run:

```bash
cd /home/ldd/anki-service
./run_supabase_tailscale.sh
```

Then open on your phone:
- `${FUNNEL_URL}/supabase.html`

Note: the demo page calls anki-service at `http://<tailscale-ip>:8000`, so your phone must be on the **tailnet** (Tailscale connected) for the API calls to succeed.

## Proxy for tailscaled (optional)

If tailscaled can't stay connected to controlplane, you can configure a SOCKS5 proxy.
Example (systemd drop-in):

```bash
sudo mkdir -p /etc/systemd/system/tailscaled.service.d
sudo tee /etc/systemd/system/tailscaled.service.d/proxy.conf >/dev/null <<'EOF'
[Service]
Environment="ALL_PROXY=socks5h://127.0.0.1:5780"
Environment="HTTPS_PROXY=socks5h://127.0.0.1:5780"
Environment="HTTP_PROXY=socks5h://127.0.0.1:5780"
Environment="NO_PROXY=127.0.0.1,localhost"
EOF
sudo systemctl daemon-reload
sudo systemctl restart tailscaled
```
