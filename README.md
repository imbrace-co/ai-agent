# AI Agent

iMBrace's AI backend + web client.

- **[`server/`](server/README.md)** — Express + TypeScript API (embedding, chat agent, MCP, trace,
  data-board AI). Runs on **`:7100`**. This is the service `marketplace` / `app-gateway` call into for
  AI flows. Full documentation: [`server/README.md`](server/README.md).
- **`client/`** — Web UI (React + Vite) for the chat / agent console.

> **Open-source edition**: the server does **not** implement `assistant_apps` / `assistants`(create) /
> `guardrail` (paid features). See [server/README.md → Edition / limitations](server/README.md#edition--limitations).

---

## Requirements

- Node.js ≥ 20
- **MongoDB** on `localhost:27017`
- **Redis** on `localhost:6379` (required by the server) — run it with Docker:

```bash
docker run -d --name imbrace-redis -p 6379:6379 redis:7-alpine
```

## Installation

```bash
npm run install:all      # installs both server/ and client/
```

> `server/` uses **npm**, `client/` uses **pnpm** (see `scripts` in `package.json`).

## Configuration

The server reads **`ai-agent/.env`** (shared). Copy it from the template and edit:

```bash
cp .env.example .env
```

Minimum for local development:

```env
PORT=7100
NODE_ENV=development
REDIS_URL=redis://localhost:6379
MONGODB_URI=mongodb://localhost:27017/messagesuggestion
```

## Running

```bash
npm run dev      # runs server (:7100) + client (Vite) IN PARALLEL
```

Run each part separately:

```bash
npm run dev --prefix server     # server only  -> http://localhost:7100
cd client && pnpm start         # client only (Vite dev server)
```

Or, from the **monorepo root**, run just the server with tagged logs:

```bash
node dev-logs.mjs aiagent        # server (:7100) -> logs/combined.log
```

## Verifying

```bash
curl http://localhost:7100/health      # { "message": "OK", ... }
```

For the API list and edition notes, see [`server/README.md`](server/README.md).

## Notes & limitations

- The **open-source edition** has no `assistant_apps`, `assistants` (create) or
  `guardrail` — these are paid features. Do not stub them into ai-agent; the AI
  parts marketplace needs (`assistant_apps`/providers) belong to the separate
  Python **aiv2** service, not this repo.
- **MongoDB and Redis are required** by the server. Without Redis (event bus) or
  MongoDB (chat / message-suggestion storage) the server will not work correctly.
- **PostgreSQL is optional** — only needed for the chat-client routes
  (`AISDK_CHAT_CLIENT_POSTGRES_URL`); leave it empty if unused.
- **An LLM key is required.** The embedding/chat flows need `OPENAI_API_KEY` (or the
  matching Bedrock/provider configuration) to actually run.
- **The client (`client/`) is front-end only.** `node dev-logs.mjs aiagent` starts the
  server alone; for the UI, also run `cd client && pnpm start`.
