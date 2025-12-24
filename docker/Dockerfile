FROM cann:8.2.rc1-910b-ubuntu22.04-py3.10 AS base

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG no_proxy
ARG NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    no_proxy=${no_proxy} \
    NO_PROXY=${NO_PROXY} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN if [ -n "${http_proxy}" ] || [ -n "${https_proxy}" ] || [ -n "${HTTP_PROXY}" ] || [ -n "${HTTPS_PROXY}" ]; then \
        echo "Acquire::http::Proxy \"${http_proxy:-$HTTP_PROXY}\";"; \
        echo "Acquire::https::Proxy \"${https_proxy:-$HTTPS_PROXY}\";"; \
    fi > /etc/apt/apt.conf.d/01proxy && \
    apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        jq \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


FROM base AS deps
WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_EXTRA_INDEX_URL
RUN if [ -n "$PIP_INDEX_URL" ]; then pip config set global.index-url "$PIP_INDEX_URL"; fi && \
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then pip config set global.extra-index-url "$PIP_EXTRA_INDEX_URL"; fi

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt



FROM deps AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir cython setuptools wheel

COPY . .

RUN python setup_cython.py build_ext --inplace
# 移除源代码
RUN find core -type f -name "*.py" ! -name "__init__.py" -delete && \
    find utils -type f -name "*.py" ! -name "__init__.py" -delete && \
    find api/routes -type f -name "*.py" ! -name "__init__.py" -delete && \
    find entity -type f -name "*.py" ! -name "__init__.py" -delete

# 移除扩展模块的调试信息
RUN find core -name "*.so" -exec strip --strip-unneeded {} + || true && \
    find utils -name "*.so" -exec strip --strip-unneeded {} + || true && \
    find api/routes -name "*.so" -exec strip --strip-unneeded {} + || true && \
    find entity -name "*.so" -exec strip --strip-unneeded {} + || true

RUN mkdir -p /opt/app_encrypted && \
    cp -r api core utils entity main.py __init__.py start.sh nginx.conf /opt/app_encrypted


FROM deps AS runtime
WORKDIR /app

COPY --from=builder /opt/app_encrypted /app
COPY config.json /config.json
COPY nginx.conf /etc/nginx/nginx.conf

RUN chmod +x /app/start.sh && \
    mkdir -p /etc/nginx/conf.d && \
    mkdir -p /app/model

EXPOSE 9000

CMD ["/bin/bash","-c","/app/start.sh"]
