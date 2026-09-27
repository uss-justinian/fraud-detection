"""Live tests: a small LLM rates tasks of rising difficulty in six problem areas.
Each area's estimates must rise with difficulty. Needs the `claude` CLI; skipped otherwise.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from estimator.model import predict
from estimator.rating import rate

LADDERS = {
    "backend": [
        "Change the default page size of the /users REST endpoint from 20 to 50.",
        "Add filtering by signup date and sorting by name to the paginated /users REST endpoint, with tests.",
        "Replace offset pagination with cursor-based pagination across all 14 list endpoints of the REST API, "
        "keeping old clients working during a deprecation period, with tests and API docs.",
    ],
    "frontend": [
        "Change the colour of the primary button to #2463EB.",
        "Add client-side validation with inline error messages to the React signup form (email, password strength, "
        "matching passwords), with component tests.",
        "Migrate the React app's global state from Redux to React Query plus context across about 40 components, "
        "keeping behaviour identical and adding tests for the data-fetching hooks.",
    ],
    "data": [
        "Fix the typo in the column alias 'totla_sales' in the monthly sales SQL query.",
        "Write a SQL migration that adds a nullable 'region' column to orders and backfills it from customers, with a rollback.",
        "Design and implement an incremental ETL pipeline that deduplicates late-arriving events, handles schema "
        "changes, and backfills three years of history into the warehouse, with data quality checks.",
    ],
    "devops": [
        "Bump the Node version in the Dockerfile from 20 to 22.",
        "Add a GitHub Actions workflow that runs lint and tests on every pull request, with dependency caching.",
        "Move the service from a single VM to Kubernetes with Helm charts, zero-downtime rolling deploys, "
        "autoscaling, secrets management and staging/production environments.",
    ],
    "security": [
        "Set the session cookie's Secure and HttpOnly flags.",
        "Add rate limiting to the login endpoint (5 attempts per minute per IP and per account), with tests.",
        "Implement OAuth2 single sign-on with PKCE for three identity providers, including account linking, "
        "token refresh, revocation and an audit log, with tests.",
    ],
    "ml": [
        "Change the random seed in the training script from 42 to 7.",
        "Add early stopping and a learning-rate scheduler to the PyTorch training loop, logging metrics to TensorBoard.",
        "Build a feature store and retraining pipeline that detects data drift, retrains the fraud model, "
        "A/B tests it against production and rolls back automatically on regressions.",
    ],
}

pytestmark = pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")


@pytest.fixture(scope="module")
def estimates():
    jobs = [(area, i, task) for area, tasks in LADDERS.items() for i, task in enumerate(tasks)]
    with ThreadPoolExecutor(6) as ex:
        ratings = list(ex.map(lambda j: rate(j[2]), jobs))
    out = {area: [None] * 3 for area in LADDERS}
    for (area, i, _), r in zip(jobs, ratings):
        out[area][i] = {"rating": r, **predict(r)}
    for area, rows in out.items():
        print(area, [(r["rating"].minutes, r["rating"].files, r["p50"]["turns"], r["size"]) for r in rows])
    return out


@pytest.mark.parametrize("area", LADDERS)
def test_estimates_rise_with_difficulty(estimates, area):
    turns = [r["p50"]["turns"] for r in estimates[area]]
    assert turns[0] < turns[1] < turns[2], turns


@pytest.mark.parametrize("area", LADDERS)
def test_easy_and_hard_land_in_sensible_sizes(estimates, area):
    easy, _, hard = estimates[area]
    assert easy["size"] in ("trivial", "small")
    assert hard["size"] in ("medium", "large", "very large")
    assert hard["p50"]["cost_usd"] > 2 * easy["p50"]["cost_usd"]
