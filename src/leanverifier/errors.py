"""Typed errors and CLI exit codes."""

from __future__ import annotations


class LeanVerifierError(Exception):
    """Base error."""

    exit_code: int = 7


class InvalidInputError(LeanVerifierError):
    exit_code = 3


class NotCertifiedError(LeanVerifierError):
    exit_code = 2


class BuildFailureError(LeanVerifierError):
    exit_code = 4


class AxiomFailureError(LeanVerifierError):
    exit_code = 5


class DigestMismatchError(LeanVerifierError):
    exit_code = 6


class TimeoutFailureError(LeanVerifierError):
    exit_code = 8


EXIT_VERIFIED = 0
EXIT_NOT_CERTIFIED = 2
EXIT_INVALID = 3
EXIT_BUILD = 4
EXIT_AXIOM = 5
EXIT_DIGEST = 6
EXIT_INTERNAL = 7
EXIT_TIMEOUT = 8
