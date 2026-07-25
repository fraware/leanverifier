from __future__ import annotations

import json
from pathlib import Path

from leanverifier.cli import main
from leanverifier.errors import EXIT_DIGEST, EXIT_NOT_CERTIFIED
from leanverifier.generator import render_module
from leanverifier.manifest import parse_model, parse_request
from tests.fixtures.builders import make_model, make_request


def test_mutation_weight_breaks_digest(tmp_path: Path) -> None:
    model = make_model([1, 1, -2], 0)
    req = make_request(model, [2, 0, 0], "1/10")
    model["weights"][0] = "2"
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    out = tmp_path / "evidence"
    mp.write_text(json.dumps(model), encoding="utf-8")
    rp.write_text(json.dumps(req), encoding="utf-8")
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_DIGEST
    )


def test_mutation_epsilon_can_uncertify(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [1], "2")
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    out = tmp_path / "evidence"
    mp.write_text(json.dumps(model), encoding="utf-8")
    rp.write_text(json.dumps(req), encoding="utf-8")
    code = main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
    assert code == EXIT_NOT_CERTIFIED


def test_mutation_comment_not_injected_as_lean() -> None:
    model = parse_model(make_model([1], 0, comment="axiom Foo : True; sorry"))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    _name, source = render_module(model, req)
    assert "axiom Foo" not in source
    assert "sorry" not in source


def test_mutation_import_removal_breaks_template() -> None:
    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    _name, source = render_module(model, req)
    mutated = source.replace("import LeanVerifier.Theorem.AffineMarginRobustness", "")
    assert "import LeanVerifier" not in mutated
