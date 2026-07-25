"""Hard resource limits for manifests, rationals, and generated artifacts.

Enforced before JSON decode where applicable. Documented in docs/model-format.md.
"""

from __future__ import annotations

# F-011 resource bounds (stable; do not raise without a findings update).
MAX_MANIFEST_BYTES = 1 * 1024 * 1024  # 1 MiB
MAX_DIMENSION = 4096
MAX_RATIONAL_CHARS = 256
MAX_COMMENT_CHARS = 512
MAX_GENERATED_LEAN_BYTES = 8 * 1024 * 1024  # 8 MiB
MAX_BUILD_LOG_BYTES = 16 * 1024 * 1024  # 16 MiB

# Stable machine-readable failure codes (InvalidInputError.code / TimeoutFailureError).
CODE_INVALID_INPUT = "invalid-input"
CODE_RESOURCE_MANIFEST_BYTES = "resource-limit-manifest-bytes"
CODE_RESOURCE_DIMENSION = "resource-limit-dimension"
CODE_RESOURCE_RATIONAL_CHARS = "resource-limit-rational-chars"
CODE_RESOURCE_COMMENT_CHARS = "resource-limit-comment-chars"
CODE_RESOURCE_GENERATED_LEAN = "resource-limit-generated-lean"
CODE_RESOURCE_BUILD_LOG = "resource-limit-build-log"
CODE_SYMLINK_REJECTED = "symlink-rejected"
CODE_UTF8_INVALID = "utf8-invalid"
CODE_TIMEOUT = "timeout"
