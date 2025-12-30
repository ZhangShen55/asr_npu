#!/bin/bash
set -euo pipefail

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0
BASE_CONFIG_PATH="${CONFIG_PATH:-/config.json}"
APP_MODULE="${APP_MODULE:-main:app}"
BASE_PORT="${BASE_PORT:-8000}"
NGINX_UPSTREAM_CONF="/etc/nginx/conf.d/backend_upstream.conf"
TMP_CONF_DIR="/tmp/app_configs"

export PYTHONPATH="/"

# 关闭 nounset，避免 set_env.sh 引用未定义变量导致退出
set +u
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # 确保 Ascend 运行时环境变量已加载
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

if [ ! -f "$BASE_CONFIG_PATH" ]; then
    echo "[ERROR] 配置文件不存在: $BASE_CONFIG_PATH"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "[ERROR] jq 未安装，无法解析配置"
    exit 1
fi

mkdir -p "$TMP_CONF_DIR"
mkdir -p /etc/nginx/conf.d

declare -a INSTANCE_PORTS=()
declare -a INSTANCE_CONFIGS=()
instance_index=0

has_npu_plan="$(jq -r 'has("npu_plan") and (.npu_plan | type == "object") and (.npu_plan | length > 0)' "$BASE_CONFIG_PATH")"

if [ "$has_npu_plan" = "true" ]; then
    mapfile -t npu_entries < <(jq -r '.npu_plan | to_entries | sort_by(.key|tonumber)[] | "\(.key)=\(.value)"' "$BASE_CONFIG_PATH")
    for entry in "${npu_entries[@]}"; do
        npu_id="${entry%%=*}"
        count="${entry#*=}"
        if ! [[ "$count" =~ ^[0-9]+$ ]]; then
            echo "[WARN] npu_plan 中实例数非法: npu_id=$npu_id, count=$count，已跳过"
            continue
        fi
        if [ "$count" -le 0 ]; then
            continue
        fi
        for ((i=0; i<count; i++)); do
            cfg_path="${TMP_CONF_DIR}/config_npu${npu_id}_${i}.json"
            jq --arg dev "npu:${npu_id}" '(.device=$dev) | del(.npu_plan)' "$BASE_CONFIG_PATH" > "$cfg_path"
            INSTANCE_CONFIGS+=("$cfg_path")
            INSTANCE_PORTS+=("$((BASE_PORT + instance_index))")
            instance_index=$((instance_index + 1))
        done
    done
else
    instance_count="$(jq -r '.instance_count // 1' "$BASE_CONFIG_PATH")"
    if ! [[ "$instance_count" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] instance_count 非法，使用默认值 1"
        instance_count=1
    fi
    device="$(jq -r '.device // "npu:0"' "$BASE_CONFIG_PATH")"
    for ((i=0; i<instance_count; i++)); do
        cfg_path="${TMP_CONF_DIR}/config_single_${i}.json"
        jq --arg dev "$device" '(.device=$dev) | del(.npu_plan)' "$BASE_CONFIG_PATH" > "$cfg_path"
        INSTANCE_CONFIGS+=("$cfg_path")
        INSTANCE_PORTS+=("$((BASE_PORT + i))")
    done
fi

if [ "${#INSTANCE_PORTS[@]}" -eq 0 ]; then
    echo "[ERROR] 未生成任何实例配置，请检查 config.json 的 npu_plan 或 instance_count"
    exit 1
fi

echo "[INFO] 启动 ${#INSTANCE_PORTS[@]} 个 uvicorn 实例..."

echo "[INFO] 生成 Nginx upstream 配置：$NGINX_UPSTREAM_CONF"
{
    echo "upstream backend {"
    echo "    least_conn;"
    for port in "${INSTANCE_PORTS[@]}"; do
        echo "    server 127.0.0.1:${port};"
    done
    echo "}"
} > "$NGINX_UPSTREAM_CONF"

if nginx -t; then
    if pgrep -x "nginx" > /dev/null; then
        nginx -s reload
    else
        nginx
    fi
else
    echo "[ERROR] Nginx 配置有误，启动失败"
    exit 1
fi

monitor_and_restart() {
    local port=$1
    local cfg=$2
    while true; do
        echo "[INFO] 启动服务实例，端口: $port, 配置: $cfg"
        CONFIG_PATH="$cfg" uvicorn "$APP_MODULE" --host 127.0.0.1 --port "$port" --workers 1
        echo "[WARN] 实例端口 $port 退出，1 秒后重启..."
        sleep 1
    done
}

for i in "${!INSTANCE_PORTS[@]}"; do
    monitor_and_restart "${INSTANCE_PORTS[$i]}" "${INSTANCE_CONFIGS[$i]}" &
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
