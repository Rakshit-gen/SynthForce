# SynthForce

A simulated startup team made of LLM agents. You describe a scenario ("we have 8 weeks to ship a mobile app for small clinics") and seven role agents talk it through turn by turn: CEO, product manager, engineering lead, designer, sales, support and a simulation analyst. Each turn ends with a coordinator summary, a task list, a function point estimate and a project timeline.

You can also ask "what if": change something about the scenario (budget cut, a key engineer leaves) and every agent says how it changes their part of the plan.

The front end lives in [SynthForce-fe](https://github.com/Rakshit-gen/SynthForce-fe). This repo is the FastAPI backend.

## How a turn works

```
POST /simulate/next
  |
  |  load session, previous summary and the last few turns from Postgres
  v
for each active agent, highest priority first
  (CEO 100, PM 80, Eng lead 75, Designer 70, Sales 65, Support 60, Analyst 50):
    - pull up to 5 of its stored memories (ranked by importance, then use)
    - add what the agents before it said this turn
    - call Groq with the role's prompt template
    - parse the reply: text, plus any tasks it proposes
  |
  v
coordinator summary  (one more LLM call over all replies)
  |
  v
function point analysis   (IFPUG weights, computed in Python from the text and tasks)
project timeline          (dependency-ordered schedule, 8h/day per agent, critical path)
  |
  v
save turn, agent replies and new memories -> return everything
```

Agents speak one after another, not in parallel, so later roles react to earlier ones in the same turn. The function point and scheduling steps are plain code, not LLM output.

What-if analysis (`POST /simulate/what-if`) asks each agent in the session to judge a list of changes against the current turn. Their confidence scores are averaged and turned into a predicted outcome and a recommendation.

## Quickstart

You need Docker and at least one [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/Rakshit-gen/SynthForce.git
cd SynthForce
cp .env.example .env     # set GROQ_API_KEY
docker-compose up --build
```

That starts the API on port 8000 plus Postgres, Redis, pgAdmin and Redis Commander. Tables are created on startup.

Start a session:

```bash
curl -X POST http://localhost:8000/simulate/start \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Launch a B2B invoicing app for freelancers in 8 weeks with 4 engineers",
       "config": {"max_turns": 10}}'
```

Then run turns with the `session_id` you got back:

```bash
curl -X POST http://localhost:8000/simulate/next \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "user_input": "The CEO wants a beta in 4 weeks"}'
```

Interactive docs are at http://localhost:8000/docs.

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/simulate/start` | Create a session. Body: `scenario`, optional `name`, `description`, `config` |
| `POST` | `/simulate/next` | Run one turn. Optional `user_input` (up to 2,000 chars) and a subset of agents to focus on |
| `POST` | `/simulate/what-if` | Judge a list of `modifications` against a turn |
| `GET` | `/simulate/{session_id}` | Session state |
| `POST` | `/simulate/{session_id}/pause` and `/resume` | Pause or resume |
| `DELETE` | `/simulate/{session_id}` | Delete a session |
| `GET` | `/simulations` | List sessions |
| `GET` | `/agents/list`, `/agents/{role}` | Role definitions |
| `GET`, `DELETE` | `/memory/{session_id}` | Read or clear stored agent memories |
| `GET` | `/health`, `/ready`, `/metrics` | Liveness, readiness, Prometheus metrics |

Session `config` fields: `max_turns` (1 to 500, default 50), `agents` (which roles take part), `temperature`, `enable_memory`, `enable_analytics`.

## Configuration

All settings come from environment variables. The full list with comments is in [`.env.example`](.env.example). The ones that matter most:

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | none | Groq key. You can also set `GROQ_API_KEY1`, `GROQ_API_KEY2`, `GROQ_API_KEY3`; the client rotates between them and backs off a key for a while after it gets rate limited |
| `DATABASE_URL` | local Postgres | Async SQLAlchemy URL (`postgresql+asyncpg://...`) |
| `REDIS_URL` | `redis://localhost:6379/0` | Storage for the rate limiter |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model for every agent |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW` | 100 / 60 | Requests per client per window |
| `MAX_SIMULATION_TURNS` | 50 | Hard cap per session |

## Layout

```
app/main.py                       app setup, middleware, health and metrics
app/core/simulation.py            HTTP routes
app/services/simulation_service.py turn loop, what-if, memory writes
app/services/groq_client.py       Groq client with key rotation and retries
app/services/function_point_analyzer.py
app/services/project_scheduler.py
app/agents/roles.py               the seven roles and their priorities
app/agents/templates.py           prompt templates
app/agents/base.py                agent class, reply parsing
app/db/                           SQLAlchemy models and repositories
infra/aws, infra/gcp              ECS and Cloud Run deploy scripts
DEPLOYMENT.md                     deploy guide
```

Unit tests: `pytest tests/unit`.

## Why these choices

- **Postgres over a document store.** Sessions, turns, agent state and memories are related rows that change together each turn, so transactions matter more than schema freedom. Flexible bits (agent replies, config) are JSONB.
- **Sequential agents.** Running them in parallel would be faster, but the point is that the engineering lead hears what the PM just said.
- **Function points and schedules in code.** Asking the model for numbers gives different numbers every run. The model proposes tasks; code does the arithmetic.

## Limits

- A turn with all seven agents is eight LLM calls. On Groq's free tier, expect rate limits after a few turns unless you add more keys.
- Memory ranking is by stored importance and access count, not by similarity to the current topic.
- Task effort comes from the model and defaults to 8 hours when it does not give one, so timelines are rough.
- There is no user auth. Every session is visible to anyone who can reach the API.
