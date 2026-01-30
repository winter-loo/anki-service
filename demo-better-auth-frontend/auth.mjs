import Database from "better-sqlite3";
import { betterAuth } from "better-auth";

// Minimal Better Auth config.
// Docs: https://better-auth.com/docs
//
// Required env vars:
// - BETTER_AUTH_SECRET (>=32 chars)
// - BETTER_AUTH_URL (e.g. http://localhost:3000)
//
// This demo uses SQLite (file) for ease.
const db = new Database("./auth.sqlite");

const baseURL = process.env.BETTER_AUTH_URL || "http://localhost:3000";
const secret = process.env.BETTER_AUTH_SECRET;
if (!secret || secret.length < 32) {
  throw new Error("BETTER_AUTH_SECRET must be set and at least 32 characters long");
}

export const auth = betterAuth({
  baseURL,
  secret,

  database: db,
  emailAndPassword: { enabled: true },

  // In production, lock this down.
  trustedOrigins: [baseURL],
});
