"""Execution sandbox package (Phase 6a, Blueprint Layer 5).

PROTECTED (research_contract.yaml protected_globs: sandbox/**).

Isolates the EXECUTION of the untrusted candidate trainer; scoring stays in the
trusted root evaluator. The evaluator loads runner.py by absolute file path
(never via sys.path); the orchestrator imports these symbols for its contract
wiring and fail-closed preflight. See runner.py for the closure rules.
"""

from sandbox.runner import (
    SUPPORTED_BACKENDS,
    ContainerSandbox,
    Sandbox,
    SandboxConfig,
    SandboxError,
    SubprocessSandbox,
    TrainResult,
    build_sandbox,
    preflight,
)

__all__ = [
    "SUPPORTED_BACKENDS",
    "ContainerSandbox",
    "Sandbox",
    "SandboxConfig",
    "SandboxError",
    "SubprocessSandbox",
    "TrainResult",
    "build_sandbox",
    "preflight",
]
