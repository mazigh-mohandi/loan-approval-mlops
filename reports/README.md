# Demo Video

## Build, Deployment & App Demo

> Link to the demo video (to be added before final submission):

**Video link:** `[ADD VIDEO LINK HERE]`

### What the video covers

1. **CI pipeline** — GitHub Actions triggered on push, test matrix (Python 3.11 / 3.12)
2. **Docker build** — `docker build -t loan-approval .` using `uv sync`
3. **App startup** — `docker run -p 8000:8000 loan-approval`, container logs showing model loaded
4. **FastAPI endpoints:**
   - `GET /` — health check
   - `GET /metrics` — request counters
   - `POST /predict` — example loan application request and response
