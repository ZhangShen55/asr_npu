from datetime import timedelta, datetime
from fastapi import APIRouter, Request

from app.core.config import settings
from app.utils.asr_stats import read_stats, clear_processing_tasks

router = APIRouter()


def _format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    days = total_seconds // (3600 * 24)
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    return f"{days}天 {hours}小时 {minutes}分 {seconds}秒"


@router.get("/get_status")
async def get_status(request: Request):
    stats = read_stats()
    run_start_time: datetime = request.app.state.run_start_time
    run_time = datetime.utcnow() - run_start_time

    return {
        "id_engine": settings.id_engine,
        "status": "living",
        "appVersion": settings.version,
        "nowTime": (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
        "runTime": _format_timedelta(run_time),
        "totalHaveDoneProcessTasks": stats["offline"] + stats["online"],
        "totalFailedTasks": stats["total_failed_tasks"],
        "offlineDone": stats["offline"],
        "onlineDone": stats["online"],
        "queuedTaskList": stats["queued_task_ids"],
        "processingTaskList": stats["processing_task_ids"],
    }


@router.delete("/clear_tasks_list")
async def clear_tasks_list():
    clear_processing_tasks()
    return {"msg": "正在处理的任务列表已清空", "code": 0}
