# asr_stats.py
import os
import json
import fcntl

STATS_FILE = "./asr_stats.json"

def init_stats_file():
    """确保 stats 文件存在，并初始化字段"""
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, "w") as f:
            json.dump({"offline": 0, "online": 0, "queued_task_ids": [],"processing_task_ids": [],"total_failed_tasks": 0}, f)

def reset_stats():
    """重置 stats 文件中的计数"""
    with open(STATS_FILE, "w") as f:
        json.dump({"offline": 0, "online": 0, "queued_task_ids": [],"processing_task_ids": [],"total_failed_tasks": 0}, f)

def update_stat(key: str, inc: int = 1):
    """更新某项统计计数"""
    init_stats_file()
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stats = json.load(f)
        stats[key] = stats.get(key, 0) + inc
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)

def read_stats():
    """读取当前统计数据"""
    init_stats_file()
    with open(STATS_FILE, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        stats = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return stats


def add_processing_task(task_id: str):
    init_stats_file()
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stats = json.load(f)
        if "processing_task_ids" not in stats:
            stats["processing_task_ids"] = []
        if task_id not in stats["processing_task_ids"]:
            stats["processing_task_ids"].append(task_id)
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)

def remove_processing_task(task_id: str):
    init_stats_file()
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stats = json.load(f)
        if "processing_task_ids" in stats and task_id in stats["processing_task_ids"]:
            stats["processing_task_ids"].remove(task_id)
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
        
        
def clear_processing_tasks():
    """清空正在处理的任务列表"""
    stats = read_stats()
    stats["processing_task_ids"] = []  # 清空列表
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)



def update_processing_tasks(task_id: str, add: bool = True):
    init_stats_file()
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stats = json.load(f)
        if add:
            stats["processing_task_ids"].append(task_id)
        else:
            if task_id in stats["processing_task_ids"]:
                stats["processing_task_ids"].remove(task_id)
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
        
        
        
def add_queued_task(task_id: str):
    init_stats_file()
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stats = json.load(f)
        if "queued_task_ids" not in stats:
            stats["queued_task_ids"] = []
        if task_id not in stats["queued_task_ids"]:
            stats["queued_task_ids"].append(task_id)
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
        
def remove_queued_task(task_id: str):
    init_stats_file()
    with open(STATS_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        stats = json.load(f)
        if "queued_task_ids" in stats and task_id in stats["queued_task_ids"]:
            stats["queued_task_ids"].remove(task_id)
        f.seek(0)
        json.dump(stats, f)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
        
        
def update_fail_task(inc: int = 1):
    update_stat("total_failed_tasks", inc)

        

