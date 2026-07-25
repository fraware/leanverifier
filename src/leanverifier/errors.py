"""Typed errors and CLI exit codes."""

from __future__ import annotations

from leanverifier.limits import CODE_INVALID_INPUT, CODE_TIMEOUT


class LeanVerifierError(Exception):
    """Base error."""

    exit_code: int = 7
    code: str = "internal"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class InvalidInputError(LeanVerifierError):
    exit_code = 3
    code = CODE_INVALID_INPUT

    def __init__(self, message: str = "", *, code: str = CODE_INVALID_INPUT) -> None:
        super().__init__(message, code=code)


class NotCertifiedError(LeanVerifierError):
    exit_code = 2
    code = "not-certified"


class BuildFailureError(LeanVerifierError):
    exit_code = 4
    code = "build-failure"


class AxiomFailureError(LeanVerifierError):
    exit_code = 5
    code = "axiom-failure"


class DigestMismatchError(LeanVerifierError):
    exit_code = 6
    code = "digest-mismatch"


class TimeoutFailureError(LeanVerifierError):
    exit_code = 8
    code = CODE_TIMEOUT

    def __init__(self, message: str = "", *, code: str = CODE_TIMEOUT) -> None:
        super().__init__(message, code=code)


EXIT_VERIFIED = 0
EXIT_NOT_CERTIFIED = 2
EXIT_INVALID = 3
EXIT_BUILD = 4
EXIT_AXIOM = 5
EXIT_DIGEST = 6
EXIT_INTERNAL = 7
EXIT_TIMEOUT = 8
