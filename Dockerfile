# Stage 1: Build the application (runs natively on build machine — no QEMU needed)
FROM --platform=$BUILDPLATFORM node:20-alpine AS builder

# Install pnpm globally (pin to specific version for reproducibility)
RUN npm install -g pnpm@9.12.0

WORKDIR /app

# Increase Node.js heap size to prevent OOM during client build (Vite bundling)
ENV NODE_OPTIONS="--max-old-space-size=4096"

# Copy package files for better Docker layer caching
COPY package*.json ./
COPY server/package*.json ./server/
COPY client/package*.json ./client/

# Copy pnpm lockfile for better caching
COPY client/pnpm-lock.yaml ./client/

# @imbrace/ui is published on the public npm registry, so no auth is needed.

# Install dependencies for server and client (separate RUN commands for better caching)
# No lockfile in Docker context (see .dockerignore) so npm resolves fresh and skips
# platform-incompatible optional deps (e.g. duckdb x64 binding on arm64) with a warning
RUN npm install --prefix server --legacy-peer-deps
RUN cd client && pnpm install --frozen-lockfile

# Copy all source code
COPY . .

# Build server and client
RUN npm run build

# Stage 2: Production runtime (targets the requested platform — arm64 or amd64)
FROM node:20-slim AS production

# Install system dependencies including DuckDB CLI
RUN apt-get update && apt-get install -y \
    dumb-init \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install DuckDB CLI — detect arch at runtime via uname -m (works with plain docker build and docker buildx)
# uname -m returns "aarch64" on arm64, "x86_64" on amd64
RUN DUCKDB_VERSION=$(curl -s https://api.github.com/repos/duckdb/duckdb/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/') && \
    case "$(uname -m)" in \
      aarch64) DUCKDB_ARCH="arm64" ;; \
      *)       DUCKDB_ARCH="amd64"   ;; \
    esac && \
    wget -q "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb_cli-linux-${DUCKDB_ARCH}.zip" -O /tmp/duckdb.zip && \
    unzip /tmp/duckdb.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/duckdb && \
    rm /tmp/duckdb.zip && \
    duckdb -version

# Install uv (the Python package runner) globally
RUN pip3 install --break-system-packages uv

# Debug: Verify uv installation
RUN which uv && echo "uv found in PATH"

RUN groupadd -g 1001 nodejs
RUN useradd -r -u 1001 -g nodejs nodeuser

WORKDIR /app

# Copy built server and client from builder
COPY --from=builder /app/server/dist ./server/dist
COPY --from=builder /app/client/dist ./server/client/dist
COPY --from=builder /app/server/package*.json ./server/
COPY --from=builder /app/package*.json ./

# Install only production dependencies
# npm runs on the target platform here, so it installs the correct native bindings automatically
RUN npm install --prefix server --omit=dev --legacy-peer-deps

EXPOSE 3000

# Healthcheck (update or remove as needed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node server/dist/healthcheck.js || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server/dist/index.js"]
