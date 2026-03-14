import os

# Override at runtime via the BACKEND_ENDPOINT env var (set in Cloud Run).
# Falls back to the existing deployed backend URL.
ENDPOINT = os.environ.get(
    "BACKEND_ENDPOINT",
    "https://backend-els-720685387106.us-central1.run.app"
)
