# Canonical model / request format

## Rational grammar

Rationals are JSON strings matching:

```
0
-?[1-9][0-9]*
-?[1-9][0-9]*/[1-9][0-9]*
```

Normalization rules (enforced by `leanverifier.canonicalize`):

- Denominator positive
- Fraction in lowest terms (GCD) via exact `Fraction` reduction — never `limit_denominator`
- Prefer `0` for zero; never `0/1` or `-0`
- Never emit `/1` for integers
- No `+` prefix, no leading zeros, no decimals (`3.0` rejected), no `1/0`, no `2/4`
- At most 256 characters per rational string

`epsilon` must be nonnegative (no leading `-`).

## Resource limits

Enforced before JSON decode for file size; other bounds apply at schema/parse time.
Stable failure codes are reported as `code: message` on stderr / in invalid-input results.

| Bound | Limit | Failure code |
|-------|-------|----------------|
| Manifest file size | 1 MiB | `resource-limit-manifest-bytes` |
| Dimension | 1 … 4096 | `resource-limit-dimension` |
| Rational string length | ≤ 256 chars | `resource-limit-rational-chars` |
| Comment length | ≤ 512 chars | `resource-limit-comment-chars` |
| Generated Lean source | ≤ 8 MiB | `resource-limit-generated-lean` |
| Retained build/subprocess log | ≤ 16 MiB | (truncated; timeout uses `timeout`) |
| Malformed UTF-8 | rejected | `utf8-invalid` |
| Symlink in output path | rejected | `symlink-rejected` |

Dimension and vector `minItems` are **≥ 1**. Zero-dimensional models and empty vectors are rejected; `Fin 0` is not a supported generator path.

## Canonical JSON bytes

1. UTF-8 encoding
2. Reject duplicate keys at parse time
3. Object keys sorted lexicographically
4. No insignificant whitespace (separators `,` and `:` only)
5. Digest = `sha256:` + hex of canonical bytes

Semantic difference ⇒ different digest. Key reorder ⇒ same digest. Weight reorder ⇒ different digest.

## Schemas

Runtime loads schemas from packaged `leanverifier.resources` via `importlib.resources`.
The checkout `schemas/` tree is a documentation/CI mirror and must match resource hashes
(`python scripts/regen_resource_hashes.py` after edits).

| File | `schema_version` |
|------|------------------|
| `schemas/model.schema.json` (packaged twin under `resources/schemas/`) | `leanverifier.model.v1` |
| `schemas/request.schema.json` | `leanverifier.request.v1` |
| `schemas/result.schema.json` | `leanverifier.result.v1` |

All use `additionalProperties: false`. JSON numbers/floats are rejected; use strings.

## Example

See `examples/affine_binary/model.json` and `examples/affine_binary/request.json`.
The request `model_digest` must equal the canonical digest of the model file.
