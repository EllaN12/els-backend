import os

_DEFAULT_BACKEND = "https://backend-els-73ttnprkbq-uc.a.run.app"

# Override at runtime via BACKEND_ENDPOINT (Cloud Run / docker -e).
# Treat empty string as unset — Cloud Run may inject BACKEND_ENDPOINT= with no value.
ENDPOINT = (os.environ.get("BACKEND_ENDPOINT") or _DEFAULT_BACKEND).rstrip("/")
