# The similaripy-server image (server-release.yml): the release binaries as
# built and tested there, nothing compiled here. Build context: a folder with
# amd64/similaripy-server and arm64/similaripy-server.
# Not pinned to a digest: each release picks up Debian's security fixes.
FROM gcr.io/distroless/cc-debian12:nonroot
ARG TARGETARCH
COPY ${TARGETARCH}/similaripy-server /usr/local/bin/similaripy-server
LABEL org.opencontainers.image.source="https://github.com/bogliosimone/similaripy" \
      org.opencontainers.image.description="similaripy-server: item-KNN recommendations over HTTP/JSON and gRPC (experimental)" \
      org.opencontainers.image.licenses="MIT"
# Inside a container the server listens on every interface (outside: 127.0.0.1).
# Env beats a config file: change it with --server.addr or SIMILARIPY_SERVER__ADDR,
# not with [server] addr in a mounted --config file
ENV SIMILARIPY_SERVER__ADDR=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["similaripy-server"]
