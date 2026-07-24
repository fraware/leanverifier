from __future__ import annotations

from pathlib import Path

import pytest

from leanverifier.errors import InvalidInputError
from leanverifier.generator import write_module_atomic
from leanverifier.manifest import parse_model, parse_request
from tests.fixtures.builders import make_model, make_request


def test_path_escape_rejected(tmp_path: Path) -> None:
    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    with pytest.raises(InvalidInputError):
        write_module_atomic(model, req, Path("relative-not-allowed"))


def test_module_name_regex() -> None:
    from leanverifier.generator import render_module

    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    with pytest.raises(InvalidInputError):
        render_module(model, req, module_name="Evil")


INJECTION_STRINGS = [
    "../../etc/passwd",
    "';echo pwned`",
    "$(curl evil)",
    "'; import Experimental",
    "end\naxiom Bad : False\n",
]


def test_injection_strings_do_not_enter_lean_body() -> None:
    from leanverifier.generator import render_module

    for payload in INJECTION_STRINGS:
        model = parse_model(make_model([1], 0, comment=payload[:512]))
        req = parse_request(make_request(model.raw, [2], "1/10"))
        _, source = render_module(model, req)
        assert payload not in source
