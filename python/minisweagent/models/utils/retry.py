"""Retry utility for model queries."""

import logging
from collections.abc import Callable

from tenacity import (
    Retrying,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _before_sleep_sanitized(logger: logging.Logger) -> Callable:
    """Produce a retry callback that never prints provider exception text.

    Transport relays can embed request identifiers, endpoint information, or
    their complete error body in an exception's string representation.
    Tenacity's stock ``before_sleep_log`` formats that representation directly.
    The retry receipt needs only the exception class and attempt ordinal; the
    caller's structured evidence remains the authority for the safe failure
    classification.
    """

    def callback(retry_state) -> None:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        exception_name = type(exc).__name__ if exc is not None else "UnknownError"
        logger.warning(
            "retrying provider transport after %s; next attempt=%d",
            exception_name,
            retry_state.attempt_number + 1,
        )

    return callback


def retry(
    *,
    logger: logging.Logger,
    abort_exceptions: list[type[Exception]],
    max_attempts: int = 1,
    retry_predicate: Callable[[BaseException], bool] | None = None,
) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.
        retry_predicate: Optional predicate limiting retries to transient
            exceptions.  When omitted, the historical policy retries every
            exception that is not in ``abort_exceptions``.

    Returns:
        A tenacity.Retrying object.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_predicate is None:
        retry_condition = retry_if_not_exception_type(tuple(abort_exceptions))
    else:
        abort_types = tuple(abort_exceptions)
        retry_condition = retry_if_exception(
            lambda exc: (
                not isinstance(exc, abort_types) and retry_predicate(exc)
            )
        )
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=_before_sleep_sanitized(logger),
        retry=retry_condition,
    )
