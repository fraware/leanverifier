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
- Fraction in lowest terms (GCD)
- Prefer `0` for zero; never `0/1` or `-0`
- Never emit `/1` for integers
- No `+` prefix, no leading zeros, no decimals (`3.0` rejected), no `1/0`, no `2/4`

`epsilon` must be nonnegative (no leading `-`).

## Canonical JSON bytes

1. UTF-8 encoding
2. Reject duplicate keys at parse time
3. Object keys sorted lexicographically
4. No insignificant whitespace (separators `,` and `:` only)
5. Digest = `sha256:` + hex of canonical bytes

Semantic difference ⇒ different digest. Key reorder ⇒ same digest. Weight reorder ⇒ different digest.

## Schemas

| File | `schema_version` |
|------|------------------|
| `schemas/model.schema.json` | `leanverifier.model.v1` |
| `schemas/request.schema.json` | `leanverifier.request.v1` |
| `schemas/result.schema.json` | `leanverifier.result.v1` |

All use `additionalProperties: false`. JSON numbers/floats are rejected; use strings.

## Example

See `examples/affine_binary/model.json` and `examples/affine_binary/request.json`.
The request `model_digest` must equal the canonical digest of the model file.
