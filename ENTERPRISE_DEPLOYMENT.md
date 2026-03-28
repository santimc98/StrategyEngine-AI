# StrategyEngine AI - Enterprise Deployment Blueprint

## Objective

This document describes the recommended deployment pattern for selling StrategyEngine AI to companies in a professional way.

It answers:

- how the product should run inside customer infrastructure
- what should be deployed first for pilots
- what should be hardened for production
- how to explain the architecture during a sales or technical validation process

## Commercial Positioning

StrategyEngine AI should be sold as a:

**self-hosted enterprise AI appliance for data-to-model workflows**

The customer:

- deploys it in their environment
- configures API keys and execution backends from the UI
- keeps datasets, runs, logs, and artifacts under their control

## Recommended Delivery Tiers

### Tier 1 - Pilot Deployment

Recommended for:
- small and medium companies
- first proof of value
- short sales cycles

Topology:
- one app container
- mounted persistent volumes
- local worker subprocess
- local sandbox or remote sandbox gateway

Operational goal:
- simplest possible deployment that still feels like a real product

### Tier 2 - Production Self-Hosted

Recommended for:
- security-reviewed environments
- multiple users
- longer-term adoption

Topology:
- dedicated UI service
- dedicated worker service
- persistent metadata and artifact stores
- isolated sandbox gateway
- centralized logs and monitoring

Operational goal:
- production reliability, security review readiness, and auditability

### Tier 3 - Managed Private Deployment

Recommended for:
- customers who do not want to operate the platform themselves

Topology:
- same architecture as self-hosted
- operated in a dedicated tenant or dedicated cloud account

Operational goal:
- lower operational burden for the customer without weakening isolation

## Minimal Deployable Components

### 1. UI / Control Plane

Current basis:
- [app.py](C:/Users/santi/Projects/Hackathon_Gemini_Agents/app.py)

Responsibilities:
- accept inputs
- render reports
- configure providers and models
- show run history and live progress

### 2. Worker / Orchestrator

Current basis:
- [src/utils/background_worker.py](C:/Users/santi/Projects/Hackathon_Gemini_Agents/src/utils/background_worker.py)
- [src/graph/graph.py](C:/Users/santi/Projects/Hackathon_Gemini_Agents/src/graph/graph.py)

Responsibilities:
- execute runs independently from browser sessions
- coordinate the multi-agent pipeline
- persist status, logs, and final outputs

### 3. Sandbox Execution Backend

Responsibilities:
- isolate user-generated code
- enforce timeout and resource boundaries
- return files, logs, and exit status

Recommended customer-facing model:
- a standard sandbox gateway contract

Reference:
- [SANDBOX_GATEWAY.md](C:/Users/santi/Projects/Hackathon_Gemini_Agents/SANDBOX_GATEWAY.md)

### 4. Persistent Storage

Used for:
- runs
- artifacts
- PDFs
- intermediate reports
- reproducibility

### 5. Config and Secrets

Used for:
- LLM API keys
- sandbox credentials
- model routing
- execution backend configuration

Direction already aligned in product:
- configured from the UI
- not edited by end users in internal files

## Recommended Deployment Patterns

## A. Pilot Using Docker Compose

Use this when:
- the customer wants something they can run quickly
- security review is lightweight
- the team is small

Recommended shape:

```text
docker-compose
  - strategyengine-ui-worker
  - mounted ./runs
  - mounted ./data
  - mounted ./artifacts
```

Notes:
- acceptable for pilot
- not the final shape for larger enterprise rollouts
- background worker can remain subprocess-based at this stage

## B. Production Inside Customer Infrastructure

Use this when:
- the customer has IT/SecOps involvement
- multiple users need access
- data governance matters

Recommended shape:

```text
Load Balancer
  -> UI / Control Plane service
  -> Worker service
  -> Artifact storage
  -> Metadata DB
  -> Secret store
  -> Sandbox gateway
```

Benefits:
- cleaner scaling
- easier observability
- better separation of concerns
- easier security review

## Security Boundary Recommendation

The most important professional recommendation is:

**run the sandbox inside the customer's own infrastructure boundary**

That means:
- local execution for low-sensitivity use cases
- or remote sandbox gateway hosted by the customer

This is the clean answer when a customer asks:
- where the code runs
- where the data lives
- who pays for compute

## Reference Answer For Customers

If a customer asks "how would this integrate with our infrastructure?", the professional answer is:

1. StrategyEngine AI is deployed inside your environment as a web application plus worker.
2. Your team configures API keys and execution backend from the UI.
3. The generated code runs in an isolated sandbox, preferably inside your own cloud or internal compute boundary.
4. All runs, artifacts, and reports remain under your control.
5. The product can operate in pilot mode on Docker Compose and later evolve to a more standard production topology without changing the core product logic.

## What To Avoid In Sales Positioning

Avoid presenting it as:
- "just a Streamlit app"
- "just a desktop tool"
- "just an `.exe`"

Those framings make enterprise buyers worry about:
- maintainability
- security
- supportability
- lack of auditability

## What To Say Instead

Prefer this language:

- "self-hosted AI analytics appliance"
- "customer-controlled execution boundary"
- "UI-first configuration with encrypted secrets"
- "isolated runtime for generated ML code"
- "deployable in Docker, Kubernetes, or private cloud"

## Immediate Productization Recommendation

For the next sales-ready step, the most practical sequence is:

1. keep the current app as the pilot appliance
2. document Docker Compose deployment clearly
3. document sandbox gateway integration clearly
4. prepare a production reference architecture diagram
5. later split UI and worker into separately deployable services

## Bottom Line

The correct professional deployment model is:

- **not** a desktop `.exe` as the main offering
- **yes** a self-hosted web product with background execution and isolated sandboxing

That is the model that enterprise customers will understand, validate, and approve more easily.
