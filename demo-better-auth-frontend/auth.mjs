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

export const auth = betterAuth({
  // Use env vars by default (recommended by Better Auth).
  // baseURL: process.env.BETTER_AUTH_URL,
  // secret: process.env.BETTER_AUTH_SECRET,

  database: db,
  emailAndPassword: { enabled: true },

  // In production, lock this down.
  trustedOrigins: [process.env.BETTER_AUTH_URL || "http://localhost:3000"],
});
