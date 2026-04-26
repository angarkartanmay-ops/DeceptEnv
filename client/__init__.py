"""HTTP-only client. Never imports server.* (OpenEnv compliance line)."""
from client.decept_client import DeceptEnvClient, StepResult

__all__ = ["DeceptEnvClient", "StepResult"]
