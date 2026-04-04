FROM ubuntu:24.04

# Install build dependencies for compiling Python from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.14 from source with GIL disabled
RUN wget -q --no-check-certificate https://www.python.org/ftp/python/3.14.3/Python-3.14.3.tgz \
    && tar -xf Python-3.14.3.tgz \
    && cd Python-3.14.3 \
    && ./configure --disable-gil --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-3.14.3 Python-3.14.3.tgz

# Verify GIL is disabled
RUN python3 -c "import sys; assert not sys._is_gil_enabled(), 'GIL is still enabled!'; print('GIL disabled confirmed')"

WORKDIR /app

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src Backend

EXPOSE 3001

ENTRYPOINT [ "python3", "-m", "Backend.server" ]
