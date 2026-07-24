# Lean-only LeanVerifier image (no Flask/ML web stack).
# Elan is installed from a versioned GitHub release tarball with sha256 verification.

FROM ubuntu:24.04@sha256:72297848456d5d37d1262630108ab308d3e9ec7ed1c3286a32fe09856619a782

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/lean \
    PATH=/home/lean/.elan/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git gcc g++ xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 lean

USER lean
WORKDIR /home/lean

ARG ELAN_VERSION=v4.2.3
ARG ELAN_ARCH=x86_64-unknown-linux-gnu
ARG ELAN_TARBALL_URL=https://github.com/leanprover/elan/releases/download/${ELAN_VERSION}/elan-${ELAN_ARCH}.tar.gz
# sha256 of elan-x86_64-unknown-linux-gnu.tar.gz for v4.2.3 (computed 2026-07-24)
ARG ELAN_TARBALL_SHA256=df0b2b3a439961ffcbb3985214365ffe40f49bc871df04dff268c7d8e21ca8b2

RUN set -eux; \
    curl -fsSL "$ELAN_TARBALL_URL" -o /tmp/elan.tar.gz; \
    echo "$ELAN_TARBALL_SHA256  /tmp/elan.tar.gz" | sha256sum -c -; \
    mkdir -p /tmp/elan-extract; \
    tar -xzf /tmp/elan.tar.gz -C /tmp/elan-extract; \
    /tmp/elan-extract/elan-init -y --default-toolchain none; \
    rm -rf /tmp/elan.tar.gz /tmp/elan-extract

WORKDIR /workspace
COPY --chown=lean:lean lean-toolchain lakefile.toml LeanVerifier.lean ./
COPY --chown=lean:lean LeanVerifier ./LeanVerifier
COPY --chown=lean:lean lake-manifest.json ./

RUN elan toolchain install "$(cat lean-toolchain)" \
    && lake build

CMD ["lake", "build"]
