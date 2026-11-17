# Local observability stack only (compose.yaml): the server and loadgen built
# from this checkout. Not the release image (../Dockerfile). Build context:
# the repo root (compose.yaml sets it); Dockerfile.dockerignore keeps it small.
FROM docker.io/library/rust:1.100.0-slim-bookworm AS build
WORKDIR /src
# The image's 1.100.0 as is: rust-toolchain.toml's clippy + rustfmt aren't
# needed to build, and would be downloaded on every uncached build
ENV RUSTUP_TOOLCHAIN=1.100.0
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates crates
RUN --mount=type=cache,target=/src/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release --locked -p similaripy-server --bins \
    && cp target/release/similaripy-server target/release/loadgen /usr/local/bin/

# Same base as the release image (Debian 12 glibc, like the build stage)
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /usr/local/bin/similaripy-server /usr/local/bin/loadgen /usr/local/bin/
ENV SIMILARIPY_SERVER__ADDR=0.0.0.0:8080
ENTRYPOINT ["similaripy-server"]
