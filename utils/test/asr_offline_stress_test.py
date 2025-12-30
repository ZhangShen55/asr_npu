#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8 并发 / 持续压测
POST http://10.236.2.52:8081/v1.1.8/seacraft_asr
Content-Type: multipart/form-data
字段:
    audioFile   : 音频文件
    showSpk     : true
    showEmotion : true
"""
import os
import time
import signal
import sys
import statistics
import threading
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ========== 配置 ==========
URL = "http://10.236.2.52:8081/v1.1.8/seacraft_asr"
AUDIO_PATH = r"/home/xjtu/zhangs/asr_dev/test.aac"  # 确保存在
TIMEOUT = 500
WORKERS = 1                     # 并发数
REPORT_INTERVAL = 1              # 统计打印间隔（秒）
# ==========================

should_stop = False
def _sig_handler(signum, frame):
    global should_stop
    should_stop = True
signal.signal(signal.SIGINT, _sig_handler)

# ---- 全局计数器/列表 ----
ok = fail = 0
latencies = []
counter = count(start=1)
lock = threading.Lock()

def worker() -> None:
    """单个工作线程：无限循环发请求直到 should_stop 置位"""
    global ok, fail
    session = requests.Session()
    data_payload = {"showSpk": "true", "showEmotion": "true"}

    while not should_stop:
        idx = next(counter)
        try:
            with open(AUDIO_PATH, "rb") as f:
                files = {"audioFile": (os.path.basename(AUDIO_PATH), f, "audio/aac")}
                start = time.perf_counter()
                resp = session.post(URL, files=files, data=data_payload, timeout=TIMEOUT)
                cost = time.perf_counter() - start
        except requests.RequestException as e:
            cost = time.perf_counter() - start
            with lock:
                fail += 1
            print(f"[{cost:.3f}s] #{idx} 请求失败: {e}")
            continue

        with lock:
            latencies.append(cost)
            if resp.status_code == 200:
                ok += 1
            else:
                fail += 1
        # 可根据需要把每次成功/失败详情打印关掉，避免刷屏
        # print(f"[{cost:.3f}s] #{idx} {'OK' if resp.status_code == 200 else 'FAIL'} {len(resp.content)}B")

def main() -> None:
    if not os.path.isfile(AUDIO_PATH):
        print(f"[ERROR] 音频文件不存在: {AUDIO_PATH}")
        sys.exit(1)

    print(f"[INFO] 开始 8 并发压测，URL={URL}，Ctrl-C 结束...")
    start_t = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # 提交 8 个长生命周期线程
        futures = [pool.submit(worker) for _ in range(WORKERS)]

        # 主线程：每秒打印一次实时统计
        try:
            while not should_stop:
                time.sleep(REPORT_INTERVAL)
                with lock:
                    cnt = ok + fail
                    if latencies:
                        avg_lat = statistics.fmean(latencies)
                        qps = cnt / (time.perf_counter() - start_t)
                        print(f"[实时] 总请求 {cnt}  OK {ok}  FAIL {fail}  "
                              f"QPS {qps:.2f}  平均延迟 {avg_lat:.3f}s")
        except KeyboardInterrupt:
            should_stop = True

        # 等待所有 worker 优雅退出
        for f in as_completed(futures):
            f.result()

    # ---- 最终汇总 ----
    total_t = time.perf_counter() - start_t
    total_req = ok + fail
    if total_req:
        avg_lat = statistics.fmean(latencies) if latencies else 0
        print("\n========== 压测结束 ==========")
        print(f"总请求: {total_req}  (成功: {ok}, 失败: {fail})")
        print(f"总耗时: {total_t:.3f}s")
        print(f"QPS:    {total_req / total_t:.2f}")
        print(f"平均延迟: {avg_lat:.3f}s")
        if latencies:
            print(f"P90 延迟: {statistics.quantiles(latencies, n=10)[8]:.3f}s")
    else:
        print("未发起任何请求。")

if __name__ == "__main__":
    import threading  # 放到最前亦可
    main()
