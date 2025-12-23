#!/bin/bash
export CONFIG_PATH="/config.json"
# ######排错检查
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONFAULTHANDLER=1
export MALLOC_ARENA_MAX=2
ulimit -c unlimited
##########

source /opt/conda/bin/activate seacraftasr

if [ "$(basename "$SHELL")" = "bash" ]; then
    conda_info=$(conda info --envs)
elif [ "$(basename "$SHELL")" = "zsh" ]; then
    conda_info=$(conda info --envs | sed 's/\r$//')
else
    echo "This script may not work correctly with this shell."
    exit 1
fi

if echo "$conda_info" | grep -q "seacraftasr"; then
    echo "Conda environment activated."
else
    echo "Failed to activate Conda environment."
    exit 1
fi

instance_count=$(jq -r '.instance_count // 4' "$CONFIG_PATH")
if [[ -z "$instance_count" || "$instance_count" == "null" ]]; then
    echo "[ERROR] 无法读取 instance_count，使用默认值 4"
    instance_count=4
fi

echo "[INFO] 启动 $instance_count 个 uvicorn 实例..."

NGINX_UPSTREAM_CONF="/etc/nginx/conf.d/backend_upstream.conf"
base_port=8000

echo "[INFO] 生成 Nginx upstream 配置：$NGINX_UPSTREAM_CONF"
echo "upstream backend {" > "$NGINX_UPSTREAM_CONF"
for ((i=0; i<instance_count; i++)); do
    port=$((base_port + i))
    echo "    server 127.0.0.1:$port;" >> "$NGINX_UPSTREAM_CONF"
done
echo "}" >> "$NGINX_UPSTREAM_CONF"

nginx -t && nginx
if nginx -t; then
    nginx -s reload || nginx
else
    echo "[ERROR] Nginx 配置有误，启动失败"
    exit 1
fi

monitor_and_restart() {
    local port=$1
    while true; do
        echo "[INFO] 启动服务实例，端口: $port"
        # 改为新的应用入口
        uvicorn main:app --host 127.0.0.1 --port "$port" --workers 1
        echo "[WARN] 实例端口 $port 退出，1 秒后重启..."
        sleep 1
    done
}

base_port=8000
for ((i=0; i<instance_count; i++)); do
    port=$((base_port + i))
    monitor_and_restart "$port" &
done

monitor_nginx() {
    while true; do
        if ! pgrep -x "nginx" > /dev/null; then
            echo "[WARN] Nginx 已退出，尝试重启..."
            nginx
        fi
        sleep 5
    done
}

monitor_nginx &
wait
