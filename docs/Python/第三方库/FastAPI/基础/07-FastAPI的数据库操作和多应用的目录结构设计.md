###### datetime:2022/09/24 15:52

###### author:nzb

# FastAPI的数据库操作和多应用的目录结构设计

```python
from fastapi import APIRouter, Depends, Request

"""【见coronavirus应用】SQL (Relational) Databases FastAPI的数据库操作"""

"""Bigger Applications - Multiple Files 多应用的目录结构设计"""


async def get_user_agent(request: Request):
    print(request.headers["User-Agent"])


app7 = APIRouter(
    prefix="/bigger_applications",
    tags=["第七章 FastAPI的数据库操作和多应用的目录结构设计"],  # 与run.py中的tags名称相同
    dependencies=[Depends(get_user_agent)],
    responses={200: {"description": "Good job!"}}
)


@app7.get("/bigger_applications")
async def bigger_applications():
    return {"message": "Bigger Applicatins - Multiple Files"}

```