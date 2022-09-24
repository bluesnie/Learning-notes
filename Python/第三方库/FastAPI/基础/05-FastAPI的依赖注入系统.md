###### datetime:2022/09/24 15:52

###### author:nzb

# FastAPI的依赖注入系统

```python
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

app5 = APIRouter()

"""Dependencies 创建、导入和声明依赖"""


async def common_parameters(q: Optional[str] = None, page: int = 1, limit: int = 100):
    return {"q": q, "page": page, "limit": limit}


@app5.get("/dependency01")
async def dependency01(commons: dict = Depends(common_parameters)):
    return commons


@app5.get("/dependency02")
def dependency02(commons: dict = Depends(common_parameters)):  # 可以在async def中调用def依赖，也可以在def中导入async def依赖
    return commons


"""Classes as Dependencies 类作为依赖项"""

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, page: int = 1, limit: int = 100):
        self.q = q
        self.page = page
        self.limit = limit


@app5.get("/classes_as_dependencies")
# async def classses_as_dependencies(commons: CommonQueryParams = Depends(CommonQueryParams)):
# async def classses_as_dependencies(commons: CommonQueryParams = Depends()):
async def classses_as_dependencies(commons=Depends(CommonQueryParams)):
    resp = {}
    if commons.q:
        resp.update({"q": commons.q})
    items = fake_items_db[commons.page: commons.page + commons.limit]
    resp.update({"items": items})
    return resp


"""Sub-dependencies 子依赖"""


def query(q: Optional[str] = None):
    # pass 根据参数需要的公共业务逻辑
    return q


def sub_query(q: str = Depends(query), last_query: Optional[str] = None):
    if not q:
        return last_query
    return q


@app5.get("/sub_dependency")
async def sub_dependency(final_query: str = Depends(sub_query, use_cache=True)):
    """use_cache默认是True, 表示当多个依赖有一个共同的子依赖时，每次request请求只会调用子依赖一次，多次调用将从缓存中获取"""
    return {"sub_dependency": final_query}


"""Dependencies in path operation decorators 路径操作装饰器中的多依赖"""


async def verify_token(x_token: str = Header(...)):
    """没有返回值的子依赖"""
    if x_token != "fake-user-token":
        return HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: str = Header(...)):
    """有返回值的子依赖，但是返回值不会被调用"""
    if x_key != "fake-user-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key


@app5.get("/dependency_in_path_operation", dependencies=[Depends(verify_token), Depends(verify_key)])
async def dependency_in_path_operation():
    return [{"user": "user01"}, {"user": "user02"}]


"""Global Dependencies 全局依赖"""

# app5 = APIRouter(dependencies=[Depends(verify_token), Depends(verify_key)])


"""Dependencies with yield 带yield的依赖"""


# 这个需要Python3.7才支持，Python3.6需要pip install async-exit-stack async-generator
# 以下都是伪代码
async def get_db():
    db = "db_connection"
    try:
        yield db
    finally:
        db.endswith("db_close")


async def dependency_a():
    dep_a = "generate_dep_a()"
    try:
        yield dep_a
    finally:
        dep_a.endswith("db_close")


async def dependency_b(dep_a=Depends(dependency_a)):
    dep_b = "generate_dep_b()"
    try:
        yield dep_b
    finally:
        dep_b.endswith(dep_a)  # 关闭子依赖 a


async def dependency_c(dep_b=Depends(dependency_b)):
    dep_c = "generate_dep_c()"
    try:
        yield dep_c
    finally:
        dep_c.endswith(dep_b)  # 关闭子依赖 b

```