from __future__ import annotations

import pytest

from leanverifier.errors import InvalidInputError
from leanverifier.manifest import parse_model, parse_request
from tests.fixtures.builders import make_model, make_request


def test_model_ok() -> None:
    m = parse_model(make_model([1, "-1/2"], "1/3"))
    assert m.dimension == 2


def test_negative_float_rejected() -> None:
    obj = make_model([1])
    obj["bias"] = 1.5  # type: ignore[assignment]
    with pytest.raises(InvalidInputError):
        # floats should be rejected either by schema path via loads or parse
        import json

        from leanverifier.canonicalize import loads_strict_json

        loads_strict_json(json.dumps(obj))


def test_negative_duplicate_key() -> None:
    from leanverifier.canonicalize import loads_strict_json

    with pytest.raises(InvalidInputError):
        loads_strict_json('{"a":1,"a":2}')


def test_negative_bad_rational() -> None:
    obj = make_model([1])
    obj["weights"] = ["2/4"]
    with pytest.raises(InvalidInputError):
        parse_model(obj)


def test_request_digest_binding() -> None:
    model = make_model([1, 2], 0)
    req = make_request(model, [0, 0], 0)
    parse_request(req)
    req["model_digest"] = "sha256:" + "a" * 64
    # schema ok but semantic check happens in generate/verify
    parse_request(req)
