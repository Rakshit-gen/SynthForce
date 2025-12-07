# Quick Deployment Steps

## Prerequisites
- Docker and Docker Compose installed
- Groq API keys (at least 1, 3 recommended)

## Step 1: Set Environment Variables

Create `.env` file in `synthetic-workforce-simulator/`:

```bash
# Required
GROQ_API_KEY1=your_key_1
GROQ_API_KEY2=your_key_2
GROQ_API_KEY3=your_key_3

# Database (default works with docker-compose)
DB_DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/workforce_simulator

# Security
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
CORS_ORIGINS=http://localhost:3000,https://your-frontend-domain.com
```

## Step 2: Deploy with Docker Compose

```bash
# From project root
docker-compose up -d

# Check logs
docker-compose logs -f workforce-simulator-app

# Verify
curl http://localhost:8000/health
```

## Step 3: Access API

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Common Commands

```bash
# Stop
docker-compose down

# Restart
docker-compose restart workforce-simulator-app

# View logs
docker-compose logs -f workforce-simulator-app

# Rebuild after code changes
docker-compose up -d --build
```

For detailed deployment options (AWS, GCP, Railway, etc.), see `DEPLOYMENT.md`.

