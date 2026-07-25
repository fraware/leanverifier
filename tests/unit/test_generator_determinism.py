from __future__ import annotations

from pathlib import Path

from leanverifier.generator import render_module, write_module_atomic
from leanverifier.manifest import parse_model, parse_request
from tests.fixtures.builders import make_model, make_request


def test_generator_determinism_100(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    model_obj = make_model([1, 1, -2], 0)
    req_obj = make_request(model_obj, [2, 0, 0], "1/10")
    model = parse_model(model_obj)
    request = parse_request(req_obj)
    name, source = render_module(model, request)
    assert name.startswith("GeneratedModel_")
    assert "![" not in source
    assert "Fin 3" in source
    assert "norm_num" in source
    assert "native" + "_decide" not in source
    assert "+" + "native" not in source
    digests: set[bytes] = set()
    for i in range(100):
        root = (tmp_path / f"out_{i}").resolve()
        root.mkdir()
        monkeypatch.chdir(root)
        path = write_module_atomic(model, request, root)
        digests.add(path.read_bytes())
    assert len(digests) == 1
    assert next(iter(digests)) == source.encode("utf-8")
