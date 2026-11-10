"""Explicit direct-Agent Lift route.

This module is intentionally a small routing boundary.  The direct route uses
the current project's Responses authoring adapter, while CandidateCheck,
canonical decoding, SourceReconcile, Replay, Gate1, and Gate2 remain owned by
the normal lifecycle.  It is never selected implicitly: the MCP adapter stays
the default in the workload runner.
"""

from __future__ import annotations

from .scrc import ResponsesSCRCLiftAdapter


DIRECT_AGENT_LIFT_BACKEND = "direct-agent"


class DirectAgentLiftAdapter(ResponsesSCRCLiftAdapter):
    """Author one candidate through the explicit non-MCP Agent route.

    The inherited adapter receives a read-only snapshot of the same
    provider-visible MCP Source/Harness resource authority, performs one
    stateless Responses call, and then applies the normal host-side candidate
    and source checks.  This class exists as the deliberate mode boundary so
    callers cannot accidentally select the route by provider or by importing
    a legacy project implementation.
    """


__all__ = ["DIRECT_AGENT_LIFT_BACKEND", "DirectAgentLiftAdapter"]
