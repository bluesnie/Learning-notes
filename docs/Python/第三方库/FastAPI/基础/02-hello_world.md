###### datetime:2022/09/24 15:44

###### author:nzb

# FastAPI hello world

```python
__date__ = "2022/7/6 14:58"
__doc__ = """第二章文件"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class CityInfo(BaseModel):
    province: str
    country: str
    is_affected: Optional[bool] = None  # 与 bool 的区别是可以不传，默认是 null


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.get("/city/{city}")
def result(city: str, query_string: Optional[str] = None):
    return {"city": city, "query_string": query_string}


@app.put("/city/{city}")
def result(city: str, city_info: CityInfo):
    return {"city": city, "countyr": city_info.country, "is_affected": city_info.is_affected}

# 启动命令：uvicorn hello_world:app --reload
```