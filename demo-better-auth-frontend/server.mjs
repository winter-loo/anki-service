import express from "express";
import crypto from "crypto";

import { auth } from "./auth.mjs";

const app = express();
app.use(express.json());

// Small helper endpoint so the static demo pages can discover config from env vars.
// Safe to expose: publishable key + project URL are public.
app.get("/demo-config.json", (req, res) => {
  res.json({
    supabaseUrl: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_PROJECT_URL || null,
    supabasePublishableKey:
      process.env.PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY ||
      process.env.SUPABASE_PUBLISHABLE_KEY ||
      process.env.SUPABASE_ANON_KEY ||
      null,
    ankiServiceUrl: process.env.ANKI_SERVICE_URL || "http://localhost:8000",
  });
});

app.use(express.static("public"));

// Mount Better Auth routes.
// NOTE: Exact handler helper depends on Better Auth runtime adapters.
// The generic approach is to forward requests to auth.handler(req).
app.all("/api/auth/*", async (req, res) => {
  const url = new URL(req.originalUrl, `${req.protocol}://${req.get("host")}`);

  // Convert Express req -> Fetch Request
  const headers = new Headers();
  for (const [k, v] of Object.entries(req.headers)) {
    if (typeof v === "string") headers.set(k, v);
    else if (Array.isArray(v)) headers.set(k, v.join(","));
  }

  const body = req.method === "GET" || req.method === "HEAD" ? undefined : JSON.stringify(req.body ?? {});
  const request = new Request(url.toString(), {
    method: req.method,
    headers,
    body,
  });

  const response = await auth.handler(request);

  // Copy response back to Express
  res.status(response.status);
  response.headers.forEach((value, key) => res.setHeader(key, value));
  const buf = Buffer.from(await response.arrayBuffer());
  res.send(buf);
});

function b64url(buf) {
  return buf.toString("base64").replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}

// RS256 keypair (demo-only).
// In production you'd use a stable key (HSM/KMS/etc) and rotate keys.
const kid = process.env.ANKI_JWT_KID || "demo-key-1";
const { privateKey, publicKey } = crypto.generateKeyPairSync("rsa", {
  modulusLength: 2048,
  publicKeyEncoding: { type: "spki", format: "pem" },
  privateKeyEncoding: { type: "pkcs8", format: "pem" },
});

function signRS256({ header, payload, privateKeyPem }) {
  const h = b64url(Buffer.from(JSON.stringify(header)));
  const p = b64url(Buffer.from(JSON.stringify(payload)));
  const input = `${h}.${p}`;
  const sig = crypto.createSign("RSA-SHA256").update(input).end().sign(privateKeyPem);
  return `${input}.${b64url(sig)}`;
}

// JWKS endpoint (anki-service will fetch this)
app.get("/.well-known/jwks.json", (req, res) => {
  // Node can export JWK directly.
  const jwk = crypto.createPublicKey(publicKey).export({ format: "jwk" });
  res.json({
    keys: [
      {
        kty: jwk.kty,
        n: jwk.n,
        e: jwk.e,
        kid,
        use: "sig",
        alg: "RS256",
      },
    ],
  });
});

// Mint a Bearer JWT for anki-service.
// Browser calls this after login; we validate the Better Auth session server-side.
app.get("/api/anki-token", async (req, res) => {

  // Ask Better Auth for the session using incoming cookies.
  const headers = new Headers();
  if (req.headers.cookie) headers.set("cookie", req.headers.cookie);

  const session = await auth.api.getSession({ headers });
  if (!session) return res.status(401).json({ error: "Not signed in" });

  // Use user id as tenant id.
  const sub = session.user?.id;
  if (!sub) return res.status(500).json({ error: "Session missing user id" });

  const now = Math.floor(Date.now() / 1000);
  const iss = process.env.ANKI_JWT_ISSUER || process.env.BETTER_AUTH_URL;
  const aud = process.env.ANKI_JWT_AUDIENCE;

  const token = signRS256({
    header: { alg: "RS256", typ: "JWT", kid },
    payload: {
      sub,
      iss,
      aud,
      iat: now,
      exp: now + 60 * 60, // 1h
    },
    privateKeyPem: privateKey,
  });

  res.json({ token, sub });
});

app.get("/api/ping-anki", async (req, res) => {
  const ankiBase = process.env.ANKI_SERVICE_URL || "http://localhost:8000";

  // Get a JWT bound to the logged-in user

  const headers = new Headers();
  if (req.headers.cookie) headers.set("cookie", req.headers.cookie);
  const session = await auth.api.getSession({ headers });
  if (!session) return res.status(401).json({ error: "Not signed in" });

  const now = Math.floor(Date.now() / 1000);
  const iss = process.env.ANKI_JWT_ISSUER || process.env.BETTER_AUTH_URL;
  const aud = process.env.ANKI_JWT_AUDIENCE;
  const token = signRS256({
    header: { alg: "RS256", typ: "JWT", kid },
    payload: { sub: session.user.id, iss, aud, iat: now, exp: now + 60 * 10 },
    privateKeyPem: privateKey,
  });

  const r = await fetch(`${ankiBase}/api/note/list`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const data = await r.json();
  res.status(r.status).json(data);
});

const port = Number(process.env.PORT || 3000);
app.listen(port, () => {
  console.log(`Demo frontend running at http://localhost:${port}`);
  console.log(`Auth routes: http://localhost:${port}/api/auth/*`);
});
