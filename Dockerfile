FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG NODE_MAJOR=22

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    tree \
    git \
    nano \
    g++ \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Pin OPACA Core to a deterministic commit (overridable via build args)
ARG OPACA_CORE_REF=3579d358699f01992c2e3fe6245e564939874a86
RUN git clone --no-checkout https://github.com/GT-ARC/opaca-core.git /code/opaca-core && \
    cd /code/opaca-core && \
    git fetch --depth 1 origin "${OPACA_CORE_REF}" && \
    git checkout --detach FETCH_HEAD

# Pin OPACA LLM Benchmark Containers to a deterministic commit (overridable via build args)
ARG OPACA_BENCHMARK_REF=3f5fc9811022ad8a7db6b23e5eb1d450034b461c
RUN git clone --no-checkout https://github.com/RobertStrehlow/opaca-llm-benchmark-containers.git /code/opaca-llm-benchmark-containers && \
    cd /code/opaca-llm-benchmark-containers && \
    git fetch --depth 1 origin "${OPACA_BENCHMARK_REF}" && \
    git checkout --detach FETCH_HEAD

# Copy requirements file
COPY requirements.txt /code/

# Install Python dependencies
RUN pip3 install -r /code/requirements.txt

WORKDIR /code
