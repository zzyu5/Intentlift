class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class CandidateRejected(InterruptAgentFlow):
    """Raised when one submitted candidate fails host admission checks.

    A rejected candidate ends the current logical Lift immediately.  The
    bounded CandidateCheck receipt is retained by the host for the next fresh
    relift; it is not returned as a normal MCP observation for the same agent
    to reinterpret or submit again.
    """


class LiftPreparationFailed(InterruptAgentFlow):
    """End a Lift before candidate submission on a closed MCP lifecycle error.

    This is distinct from ``CandidateRejected``: no candidate reached semantic
    admission, so a partial buffer must not become a repairable draft.
    """


class ReplayOrchestrationExceeded(InterruptAgentFlow):
    """End Replay when a bounded construction-repair lifecycle is exhausted.

    This carries no semantic acceptance implication.  In particular, it cannot
    turn a preflight rejection into an acceptable Replay source or a Gate1
    result.
    """


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class MiniSWERequestCancelled(RuntimeError):
    """Abort one in-flight agent request after another candidate has won."""

    def __init__(self, winner_branch_id: int) -> None:
        if not isinstance(winner_branch_id, int) or winner_branch_id < 1:
            raise ValueError("winner_branch_id must be positive")
        self.winner_branch_id = winner_branch_id
        super().__init__(
            f"candidate superseded by Gate1 winner branch {winner_branch_id}"
        )


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""
