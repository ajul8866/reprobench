# ReproBench agent container (paper Sec. 5 "Harness"; plan Sec. (5)).
#
# One container is created per run with `--rm`, mounting only:
#   -v <run_dir>/output:/output
#   -v <run_dir>/trace:/trace
# Reference profiles (data/profiles) are NEVER mounted into this container
# (leak-prevention invariant enforced by tests/test_no_leak.py and
# benchmark.yaml harness.mount_profiles: false).
#
# Build:
#   docker build -f docker/agent.Dockerfile -t reprobench/agent:24.04 .
#
# Optional build args:
#   INSTALL_GHIDRA=1   download a Ghidra release and install headless analyzer
#   INSTALL_OPENCODE=1 install the opencode CLI from npm (@opencode/cli)
#   GHIDRA_VERSION     Ghidra release tag (default 11.1.2)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG INSTALL_GHIDRA=1
ARG INSTALL_OPENCODE=1
ARG GHIDRA_VERSION=11.1.2

# ---------------------------------------------------------------------------
# Base toolchain required by the reproduction workflow (plan Sec. (5)).
#   binwalk, squashfs-tools, qemu-user-static, qemu-system-arm (also ships
#   qemu-system-aarch64 on Ubuntu), ghidra headless, file, binutils,
#   gdb-multiarch, strace, netcat, tcpdump, ubireader, jefferson, sasquatch,
#   python3, curl, wget, git, node + npm.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        binwalk \
        squashfs-tools \
        qemu-user-static \
        qemu-system-arm \
        qemu-utils \
        file \
        binutils \
        binutils-multiarch \
        gdb-multiarch \
        strace \
        ltrace \
        netcat-openbsd \
        tcpdump \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        wget \
        git \
        ca-certificates \
        unzip \
        xz-utils \
        bzip2 \
        gzip \
        tar \
        cpio \
        p7zip-full \
        lz4 \
        lzop \
        zstd \
        liblzma-dev \
        libssl-dev \
        libffi-dev \
        build-essential \
        cmake \
        autoconf \
        automake \
        libtool \
        pkg-config \
        sudo \
        vim-tiny \
        less \
        jq \
        procps \
        iproute2 \
        iputils-ping \
        dnsutils \
        openssh-client \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python-side firmware tooling that is not packaged in Ubuntu:
#   ubireader-ng  -> ubireader_extract_images / ubireader_extract_files
#   jefferson     -> JFFS2 extraction
#   pyelftools    -> ELF inspection used by candidate code
# sasquatch (non-standard squashfs) is optional; we try a source build but
# never fail the image when the network/compiler is unavailable.
# ---------------------------------------------------------------------------
RUN pip3 install --break-system-packages --no-cache-dir \
        ubireader-ng \
        jefferson \
        pyelftools \
        python-magic \
        httpx \
        PyYAML \
    || pip3 install --no-cache-dir ubireader-ng jefferson pyelftools python-magic httpx PyYAML

# Optional: build sasquatch (squashfs with vendor patches) from source.
RUN (git clone --depth 1 https://github.com/devttys0/sasquatch.git /tmp/sasquatch \
        && cd /tmp/sasquatch \
        && wget -q https://github.com/devttys0/sasquatch/pull/47.patch -O /tmp/sasquatch.patch || true \
        && (patch -p1 < /tmp/sasquatch.patch || true) \
        && ./build.sh) \
    && cp -a /tmp/sasquatch/sasquatch /usr/local/bin/sasquatch 2>/dev/null || true \
    && rm -rf /tmp/sasquatch /tmp/sasquatch.patch

# ---------------------------------------------------------------------------
# Ghidra headless (optional; large download).
# ---------------------------------------------------------------------------
RUN if [ "$INSTALL_GHIDRA" = "1" ]; then \
        set -eux; \
        apt-get update && apt-get install -y --no-install-recommends default-jdk-headless; \
        rm -rf /var/lib/apt/lists/*; \
        url="https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_$(date +%Y%m%d).zip"; \
        (curl -fsSL "$url" -o /tmp/ghidra.zip \
            && unzip -q /tmp/ghidra.zip -d /opt \
            && mv /opt/ghidra_* /opt/ghidra) \
        || echo "ghidra download unavailable; continuing without it"; \
        rm -f /tmp/ghidra.zip; \
    else \
        echo "skipping ghidra"; \
    fi

# ---------------------------------------------------------------------------
# opencode agent scaffold (optional at build time; the runner also supports
# the OpenAI-compatible fallback adapter).
# ---------------------------------------------------------------------------
RUN if [ "$INSTALL_OPENCODE" = "1" ]; then \
        npm install -g @opencode/cli || echo "opencode install failed; fallback adapter available"; \
    else \
        echo "skipping opencode"; \
    fi

# ---------------------------------------------------------------------------
# Non-root agent user with the paper's sudo password `000000`
# (Fig. 3 Rule 4). The password is intentionally weak and lives only inside
# the throwaway container.
# ---------------------------------------------------------------------------
RUN useradd -m -s /bin/bash -G sudo agent \
    && echo 'agent:000000' | chpasswd \
    && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/agent \
    && chmod 0440 /etc/sudoers.d/agent

# Run directories are the only writable mounts.
RUN mkdir -p /output /trace /opt/reprobench \
    && chown -R agent:agent /output /trace /opt/reprobench

# Entrypoint helper: forwards into the container command with a SIGTERM trap
# so the 24h watchdog / budget abort can stop the workload cleanly.
COPY docker/entrypoint.sh /usr/local/bin/reprobench-entrypoint
RUN chmod 0755 /usr/local/bin/reprobench-entrypoint

# Record image-side tool versions for meta.json (plan Sec. (5): "meta.json
# with image digest and tool versions").
RUN { \
        echo "image=reprobench/agent:24.04"; \
        echo "ubuntu=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)"; \
        for t in binwalk unsquashfs qemu-arm-static qemu-aarch64-static qemu-system-arm qemu-system-aarch64 file objdump gdb-multiarch strace nc tcpdump python3 node npm git; do \
            v="$(command -v "$t" >/dev/null 2>&1 && echo present || echo absent)"; \
            echo "$t=$v"; \
        done; \
    } > /opt/reprobench/tool_versions.txt

WORKDIR /output
USER agent
ENTRYPOINT ["/usr/local/bin/reprobench-entrypoint"]
CMD ["/bin/bash", "-lc", "echo 'ReproBench agent container ready'; ls -la /output /trace"]
