###### datetime:2022/09/24 15:52

###### author:nzb

# 入口文件

```python
import asyncio
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import PlainTextResponse
# from starlette.exceptions import HTTPException as StarletteHTTPException

from tutorial import app3, app4, app5, app6, app7, app8
from coronavirus import application
from timed_task import tasks_router

app = FastAPI(
    title="FastAPI tutorial and Coronavirus Tracker API Docs",
    description='FastAPI教程 新冠病毒疫情跟踪器API接口文档',
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redocs",
    # dependencies=[], # 全局依赖
)
# mount表示将某个目录下一个完全独立的应用挂载过来，这个不会在API交互文档中显示
app.mount(path="/static",
          app=StaticFiles(directory="./coronavirus/static"), name='static')  # .mount()不要在分路由APIRouter().mount()调用，模板会报错


# app.debug = True

# @app.exception_handler(StarletteHTTPException)  # 重写HTTPException异常处理器
# async def http_exception_handler(request, exc):
#     """
#     :param request: 这个参数不能省
#     :param exc:
#     :return:
#     """
#     return PlainTextResponse(str(exc.detail), status_code=exc.status_code)
#
#
# @app.exception_handler(RequestValidationError)  # 重写请求验证异常处理器
# async def validation_exception_handler(request, exc):
#     """
#     :param request: 这个参数不能省
#     :param exc:
#     :return:
#     """
#     return PlainTextResponse(str(exc), status_code=400)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    resp = await call_next(request)
    process_time = time.time() - start_time
    resp.headers['X-Process-Time'] = str(process_time)
    return resp


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(app3, prefix="/chapter03", tags=["第三章 请求参数和验证"])
app.include_router(app4, prefix="/chapter04", tags=["第四章 响应处理和FastAPI配置"])
app.include_router(app5, prefix="/chapter05", tags=["第五章 FastAPI的依赖注入系统"])
app.include_router(app6, prefix="/chapter06", tags=["第六章 安全、认证和授权"])
app.include_router(app7, prefix="/chapter07", tags=['第七章 FastAPI的数据库操作和多应用的目录结构设计'])
app.include_router(app8, prefix='/chapter08', tags=['第八章 中间件、CORS、后台任务、测试用例'])
app.include_router(application, prefix="/coronavirus", tags=["新冠病毒疫情跟踪器API"])
app.include_router(tasks_router, prefix="/timed_tasks", tags=["apSheduler动态定时任务"])

if __name__ == '__main__':
    threading.Thread(target=lambda: asyncio.new_event_loop().run_forever).start()
    uvicorn.run('main:app', host='127.0.0.1', port=8001, reload=True, debug=True, workers=5)

```