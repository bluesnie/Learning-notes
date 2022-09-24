###### datetime:2022/09/24 15:52

###### author:nzb

# apSheduler动态定时任务

```python
__doc__ = """
http://t.zoukankan.com/CharmCode-p-14191009.html
http://t.zoukankan.com/zhangliang91-p-12468871.html
"""

import asyncio
import datetime

from enum import Enum

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, Body, Query

tasks_router = APIRouter()

scheduler = AsyncIOScheduler(timezone='Asia/Shanghai')


def print_time(name):
    print(f'{name} - {datetime.datetime.now()}')


async def tick(num):
    await asyncio.sleep(1)
    print(f'Tick{num}! The time is: %s' % datetime.datetime.now())


@tasks_router.on_event("startup")
def init_scheduler():
    """初始化定时任务调度器"""
    jobstores = {
        # 'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')  # SQLAlchemyJobStore指定存储链接
        'default': MemoryJobStore()  # SQLAlchemyJobStore指定存储链接
    }
    executors = {
        # 'default': {'type': 'threadpool', 'max_workers': 20},  # 最大工作线程数20
        # 'default': ThreadPoolExecutor(max_workers=20)  # 最大工作进程数为5
        # 'processpool': ProcessPoolExecutor(max_workers=5)  # 最大工作进程数为5
        'default': AsyncIOExecutor()
    }
    scheduler.configure(jobstores=jobstores, executors=executors)
    # # 添加一个coroutine执行，结果很不理想...
    scheduler.add_job(func=tick, args=(1,), trigger=CronTrigger.from_crontab("* * * * *"), timezone='Asia/Shanghai',
                      next_run_time=datetime.datetime.now().astimezone())
    print("启动调度器...")
    scheduler.start()


@tasks_router.post('/get_jobs', summary="获取所有jobs")
async def get_jobs():
    """获取所有jobs"""
    res = []
    data = scheduler.get_jobs()
    for i in data:
        tmp = dict()
        tmp['id'] = i.id
        tmp['name'] = i.name
        tmp['next_run_time'] = i.next_run_time.strftime("%F %X")
        tmp['timezone'] = str(i.next_run_time.tzinfo)
        res.append(tmp)
    return {"msg": "success!", "data": res}


@tasks_router.post('/print_jobs', summary="打印jobs")
async def print_jobs():
    """打印jobs"""
    scheduler.print_jobs()
    return {"msg": "success!"}


@tasks_router.post('/add_job', summary="添加job")
async def add_job(job_id: str = Body(...), cron: str = Body(...)):
    """添加job"""
    # scheduler.add_job(id=job_id, func=print_time, args=(job_id,), trigger=CronTrigger.from_crontab(cron))
    # scheduler.add_job(id=job_id, func=print_time, args=(job_id,), trigger=IntervalTrigger(seconds=3))
    scheduler.add_job(id=job_id, func=tick, args=(job_id,), trigger=IntervalTrigger(seconds=3))
    return {"msg": "success!"}


class TriggerTypeEnum(str, Enum):
    """触发类型枚举"""
    CRON = "cron"
    INTERVAL = "interval"


@tasks_router.post('/modify_job', summary="修改job")
async def modify_job(
        job_id: str = Query(..., description="job id"),
        trigger_type: TriggerTypeEnum = Query(default=TriggerTypeEnum.INTERVAL,
                                              description="interval: 固定时间间隔运行job "
                                                          "<br > cron: 类似linux-crontab，某个时间点定期运行job,"
                                                          "帮助网站：https://crontab.guru"),
        interval_seconds: int = Query(None, description="IntervalTrigger 的秒数"),
        cron_exp: str = Query(None, description="CronTrigger 表达式"),
):
    """修改job"""
    if trigger_type == TriggerTypeEnum.INTERVAL:
        trigger = IntervalTrigger(seconds=interval_seconds)
    else:
        trigger = CronTrigger.from_crontab(cron_exp)
    ntime = datetime.datetime.now()
    next_time = trigger.get_next_fire_time(ntime, ntime)
    scheduler.modify_job(job_id=job_id, trigger=trigger, next_run_time=next_time)
    return {"msg": "success!"}


@tasks_router.post('/pause_job', summary="停止job")
async def pause_job(job_id):
    """停止job"""
    scheduler.pause_job(job_id)
    print(f"停止job - {job_id}")


@tasks_router.post('/resume_job', summary="恢复job")
async def resume_job(job_id):
    """恢复job"""
    scheduler.resume_job(job_id)
    print(f"恢复job - {job_id}")


@tasks_router.post('/remove_job', summary="移除job")
async def remove_job(job_id: str = Body(..., embed=True)):
    """移除job"""
    scheduler.remove_job(job_id)
    return {"msg": "success!"}

```