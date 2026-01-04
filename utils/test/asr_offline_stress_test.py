#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import signal
import sys
import statistics
import threading
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ========== 核心配置调整 ==========
URL = "http://10.236.2.52:8081/v1.1.8/seacraft_asr"
AUDIO_PATH = r"/home/xjtu/zhangs/asr_dev/teacher1.wav"
# 调整 1: 平均5分钟，超时建议设为 15分钟(900s) 或更长，防止客户端因波动断开
TIMEOUT = 900
WORKERS = 16
REPORT_INTERVAL = 2  # 打印间隔不用太密
# =================================

should_stop = False


def _sig_handler(signum, frame):
    global should_stop
    should_stop = True
    print("\n[WARN] 正在停止，等待剩余请求返回...", flush=True)


signal.signal(signal.SIGINT, _sig_handler)

# 统计变量
ok = fail = 0
processing = 0  # [新增] 当前正在处理请求数
latencies = []
counter = count(start=1)
lock = threading.Lock()


def worker():
    global ok, fail, processing
    session = requests.Session()
    data_payload = {"showSpk": "true", "showEmotion": "true"}

    while not should_stop:
        idx = next(counter)
        with lock:
            processing += 1  # 标记开始

        try:
            with open(AUDIO_PATH, "rb") as f:
                files = {"audioFile": (os.path.basename(AUDIO_PATH), f, "audio/aac")}
                start = time.perf_counter()

                # 这里会阻塞 5 分钟左右
                resp = session.post(URL, files=files, data=data_payload, timeout=TIMEOUT)
                cost = time.perf_counter() - start
                _ = resp.content  # 确保读完

        except Exception as e:
            cost = time.perf_counter() - start
            with lock:
                fail += 1
                processing -= 1
            print(f"[ERROR] #{idx} 耗时{cost:.1f}s 异常: {e}", flush=True)
            continue

        with lock:
            processing -= 1  # 标记结束
            latencies.append(cost)
            if resp.status_code == 200:
                ok += 1
                print(f"[SUCCESS] #{idx} 完成，耗时 {cost:.2f}s", flush=True)
            else:
                fail += 1
                print(f"[FAIL] #{idx} 状态码 {resp.status_code} 耗时 {cost:.2f}s", flush=True)


def main():
    if not os.path.isfile(AUDIO_PATH):
        print(f"文件不存在: {AUDIO_PATH}")
        sys.exit(1)

    print(f"[INFO] 启动压测 | 并发: {WORKERS} | 超时: {TIMEOUT}s | 预计首包耗时 > 5min", flush=True)
    start_t = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(worker) for _ in range(WORKERS)]

        try:
            while not should_stop:
                time.sleep(REPORT_INTERVAL)
                with lock:
                    cnt = ok + fail
                    # 计算平均耗时
                    avg_lat = statistics.fmean(latencies) if latencies else 0.0
                    run_t = time.perf_counter() - start_t
                    # QPS = 完成总数 / 运行时间 (针对长耗时接口，QPS会很低)
                    real_qps = cnt / run_t if run_t > 0 else 0

                    # [重点] 增加了 'Pending' 字段，让您知道有 16 个都在跑
                    print(
                        f"\r[运行中] 时间:{int(run_t)}s | Pending:{processing} | OK:{ok} | Fail:{fail} | AvgLatency:{avg_lat:.1f}s",
                        end="", flush=True)
        except KeyboardInterrupt:
            should_stop = True

        print("\n等待线程回收...")
        for f in as_completed(futures):
            f.result()


if __name__ == "__main__":
    main()