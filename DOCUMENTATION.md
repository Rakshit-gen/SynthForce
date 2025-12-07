# Synthetic Workforce Simulator - Complete Backend Documentation

## Project Overview

A production-grade multi-agent simulation engine where CEO, PM, Engineering Lead, Designer, Sales, Support, and Simulation Analyst agents collaborate using ultra-fast LLM inference via Groq API.

## Tech Stack Justification

### FastAPI over Node.js
- **Native async/await**: Superior for I/O-bound LLM API calls
- **Python AI ecosystem**: Direct integration with AI/ML libraries
- **Type hints + Pydantic**: Runtime validation and OpenAPI generation
- **Better Groq SDK support**: Native Python client
- **Performance**: Comparable to Node.js with uvicorn/gunicorn

### PostgreSQL over MongoDB
- **ACID compliance**: Critical for maintaining consistent agent state
- **Relational queries**: Complex queries for agent interactions
- **JSON support**: JSONB for flexible memory storage
- **Mature ecosystem**: Better tooling for production deployments

---

## Project Structure

```
synthetic-workforce-simulator/
├── app/
│   ├── api/                    # REST API endpoints
│   ├── agents/                 # Agent definitions
│   │   ├── base.py             # Base agent class + factory
│   │   ├── roles.py            # 7 agent role definitions
│   │   └── templates.py        # Prompt templates
│   ├── core/                   # Core engine
│   │   ├── agent_engine.py     # LLM wrapper + orchestration
│   │   ├── coordinator.py      # High-level coordination
│   │   └── simulation.py       # REST API controller
│   ├── db/                     # Database layer
│   │   ├── database.py         # Async SQLAlchemy setup
│   │   └── repositories.py     # Data access layer
│   ├── models/                 # Data models
│   │   ├── schemas.py          # Pydantic schemas
│   │   └── orm.py              # SQLAlchemy ORM models
│   ├── services/               # Business logic
│   │   ├── groq_client.py      # Groq API wrapper
│   │   └── simulation_service.py
│   ├── utils/                  # Utilities
│   │   ├── logging.py          # Structured logging
│   │   ├── rate_limiter.py     # Rate limiting
│   │   └── security.py         # CORS + security headers
│   ├── config.py               # Configuration management
│   └── main.py                 # Application entry point
├── infra/
│   ├── aws/                    # AWS ECS deployment
│   └── gcp/                    # GCP Cloud Run deployment
├── scripts/
│   └── init-db.sql
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Groq API Key (https://console.groq.com)

### Local Development

```bash
# Clone and setup
git clone <repository>
cd synthetic-workforce-simulator
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Start with Docker Compose
docker-compose up --build

# Access
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate/start` | POST | Start a new simulation session |
| `/simulate/next` | POST | Execute next turn in simulation |
| `/simulate/what-if` | POST | Run what-if scenario analysis |
| `/agents/list` | GET | List all available agents |
| `/memory/{sessionId}` | GET | Get memory for a session |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Example: Start Simulation

```bash
curl -X POST http://localhost:8000/simulate/start \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "We need to launch a new AI-powered feature within 3 months. Budget is $500K. Team has 8 engineers available.",
    "name": "Q2 Feature Launch",
    "config": {
      "max_turns": 20,
      "agents": ["ceo", "pm", "engineering_lead", "designer"]
    }
  }'
```

### Example: Execute Next Turn

```bash
curl -X POST http://localhost:8000/simulate/next \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "user_input": "Let focus on technical feasibility first"
  }'
```

---

## Agent Roles

| Role | Priority | Description |
|------|----------|-------------|
| CEO | 100 | Strategic leader, vision setting, major decisions |
| PM | 80 | Product strategy, roadmap, requirements |
| Engineering Lead | 75 | Technical architecture, feasibility, team capacity |
| Designer | 70 | UX/UI design, user research, design systems |
| Sales | 65 | Revenue, customer acquisition, market feedback |
| Support | 60 | Customer success, issue patterns, feedback |
| Simulation Analyst | 50 | Meta-analysis, pattern recognition, insights |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  /simulate   │  │   /agents    │  │   /memory    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Simulation Service Layer                    │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │ SimulationService│  │   Agent Coordinator         │   │   │
│  │  └────────┬────────┘  └────────────┬────────────────┘   │   │
│  └───────────┼────────────────────────┼────────────────────┘   │
│              ▼                        ▼                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Agent Engine                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │   CEO   │ │   PM    │ │  Eng    │ │Designer │ ...    │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │   │
│  │       └───────────┴───────────┴───────────┘              │   │
│  │                        │                                 │   │
│  │              ┌─────────▼─────────┐                       │   │
│  │              │   Groq Client     │                       │   │
│  │              │  (Ultra-fast LLM) │                       │   │
│  │              └───────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │                    Data Layer                            │   │
│  │  ┌────────────────┐          ┌────────────────┐         │   │
│  │  │   PostgreSQL   │          │     Redis      │         │   │
│  │  │  - Sessions    │          │  - Rate Limit  │         │   │
│  │  │  - Turns       │          │  - Cache       │         │   │
│  │  │  - Memory      │          │                │         │   │
│  │  └────────────────┘          └────────────────┘         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Agent Engine (`app/core/agent_engine.py`)
- LLM wrapper for Groq integration
- Turn-based orchestration
- Memory management
- Hooks for extensibility

### 2. Groq Client (`app/services/groq_client.py`)
- Async HTTP client with retries
- Rate limit handling
- Health checks

### 3. Simulation Service (`app/services/simulation_service.py`)
- Session management
- Turn execution
- What-if analysis

### 4. Database Models (`app/models/orm.py`)
- SimulationSession
- SimulationTurn
- AgentState
- AgentMemory
- WhatIfScenario

---

## Configuration

All configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | Required |
| `DATABASE_URL` | PostgreSQL connection | Required |
| `REDIS_URL` | Redis connection | Required |
| `ENVIRONMENT` | dev/staging/prod | development |
| `LOG_LEVEL` | Logging level | INFO |
| `RATE_LIMIT_REQUESTS` | Requests per window | 100 |
| `CORS_ORIGINS` | Allowed origins | localhost |

---

## Cloud Deployment

### AWS ECS

```bash
cd infra/aws
chmod +x deploy.sh
./deploy.sh production us-east-1
```

### GCP Cloud Run

```bash
cd infra/gcp
chmod +x deploy.sh
./deploy.sh my-project us-central1
```

---

## Performance Benchmarks

### Expected Performance (Groq)
- **Inference latency**: 50-200ms per agent response
- **Tokens/second**: 300-500 tokens/s
- **Turn execution**: 2-5 seconds for 7 agents

### Optimization Suggestions

1. **Parallel Agent Execution**: Enable `enable_parallel_execution=True` in AgentEngine
2. **Response Caching**: Cache common prompts in Redis
3. **Connection Pooling**: Pre-configured in database settings
4. **Batch Requests**: Use batch endpoints for multiple operations
5. **Async Everywhere**: All I/O operations are async

---

## Security Features

- CORS with configurable origins
- Security headers (X-Frame-Options, CSP, etc.)
- Request ID tracking
- Rate limiting (Redis-backed)
- Input sanitization
- No authentication (handled by frontend via Clerk)

---

## Monitoring

- **Health endpoint**: `/health` - Component status
- **Metrics endpoint**: `/metrics` - Prometheus format
- **Structured logging**: JSON logs with request context
- **Request timing**: X-Response-Time header

---

## License

MIT License
