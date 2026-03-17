# Demo Video

## Build, Deployment & App Demo

**Video link:** [https://drive.google.com/file/d/1rozx8BAtQfk_tn0ocJM0OW0FSvLqC5LD/view?usp=drive_link](https://drive.google.com/file/d/1rozx8BAtQfk_tn0ocJM0OW0FSvLqC5LD/view?usp=drive_link)

### What the video covers

1. **CI pipeline** — GitHub Actions triggered on push, test matrix (Python 3.11 / 3.12)
2. **Docker build** — `docker build -t loan-approval .` using `uv sync`
3. **App startup** — `docker run -p 8000:8000 loan-approval`, container logs showing model loaded
4. **FastAPI endpoints:**
   - `GET /` — health check
   - `GET /metrics` — request counters
   - `POST /predict` — example loan application request and response
