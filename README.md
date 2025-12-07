# Synthetic Workforce Simulator

A production-grade multi-agent simulation engine where CEO, PM, Engineering Lead, Designer, Sales, Support, and Simulation Analyst agents collaborate using ultra-fast LLM inference via Groq API.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Synthetic Workforce Simulator                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │     CEO     │  │     PM      │  │ Eng. Lead   │  │  Designer   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│  ┌──────┴────────────────┴────────────────┴────────────────┴──────┐        │
│  │                    Agent Coordinator                           │        │
│  │  ┌─────────────────────────────────────────────────────────┐  │        │
│  │  │              Turn-Based Orchestration Engine             │  │        │
│  │  └─────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────┬───────────────────────────────────┘        │
│                               │                                             │
│  ┌────────────────────────────┴───────────────────────────────────┐        │
│  │                      Agent Engine (LLM Layer)                   │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │        │
│  │  │ Groq Client  │  │Role Templates│  │Memory Manager│          │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │        │
│  └────────────────────────────┬───────────────────────────────────┘        │
│                               │                                             │
│  ┌────────────────────────────┴───────────────────────────────────┐        │
│  │                         Data Layer                              │        │
│  │  ┌──────────────────────┐  ┌──────────────────────────┐        │        │
│  │  │     PostgreSQL       │  │     Redis (Cache)        │        │        │
│  │  │  - Sessions          │  │  - Rate Limiting         │        │        │
│  │  │  - Agent States      │  │  - Session Cache         │        │        │
│  │  │  - Memory Store      │  │                          │        │        │
│  │  └──────────────────────┘  └──────────────────────────┘        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
- **Schema enforcement**: Ensures data integrity for simulation state

## Project Structure

```
synthetic-workforce-simulator/
├── app/
│   ├── api/                    # REST API endpoints
│   │   ├── __init__.py
│   │   ├── routes.py           # Main route definitions
│   │   ├── simulate.py         # Simulation endpoints
│   │   ├── agents.py           # Agent endpoints
│   │   └── memory.py           # Memory endpoints
│   ├── agents/                 # Agent definitions
│   │   ├── __init__.py
│   │   ├── base.py             # Base agent class
│   │   ├── roles.py            # Role definitions
│   │   └── templates.py        # Prompt templates
│   ├── core/                   # Core engine
│   │   ├── __init__.py
│   │   ├── agent_engine.py     # LLM wrapper + orchestration
│   │   ├── coordinator.py      # Agent coordinator
│   │   └── simulation.py       # Simulation controller
│   ├── db/                     # Database layer
│   │   ├── __init__.py
│   │   ├── database.py         # DB connection
│   │   ├── repositories.py     # Data access layer
│   │   └── migrations/         # Alembic migrations
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic schemas
│   │   └── orm.py              # SQLAlchemy models
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── groq_client.py      # Groq API wrapper
│   │   └── simulation_service.py
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py          # Logging configuration
│   │   ├── rate_limiter.py     # Rate limiting
│   │   └── security.py         # Security headers
│   ├── config.py               # Configuration management
│   └── main.py                 # Application entry point
├── infra/                      # Infrastructure as Code
│   ├── aws/                    # AWS ECS deployment
│   │   ├── ecs-task-definition.json
│   │   ├── cloudformation.yaml
│   │   └── deploy.sh
│   └── gcp/                    # GCP Cloud Run deployment
│       ├── cloudbuild.yaml
│       ├── service.yaml
│       └── deploy.sh
├── tests/                      # Test suites
│   ├── unit/
│   └── integration/
├── scripts/                    # Utility scripts
│   ├── seed_db.py
│   └── health_check.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Groq API Key (get from https://console.groq.com)

### Local Development

1. Clone and setup:
```bash
git clone <repository>
cd synthetic-workforce-simulator
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

2. Start with Docker Compose:
```bash
docker-compose up --build
```

3. Access the API:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate/start` | POST | Start a new simulation session |
| `/simulate/next` | POST | Execute next turn in simulation |
| `/simulate/what-if` | POST | Run what-if scenario analysis |
| `/agents/list` | GET | List all available agents |
| `/memory/{sessionId}` | GET | Get memory for a session |

## Configuration

All configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API authentication key | Required |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `LOG_LEVEL` | Logging level | INFO |
| `RATE_LIMIT_REQUESTS` | Requests per window | 100 |
| `RATE_LIMIT_WINDOW` | Window in seconds | 60 |

## Deployment

See `infra/aws/` or `infra/gcp/` for cloud deployment instructions.

## License

MIT License
# SynthForce
