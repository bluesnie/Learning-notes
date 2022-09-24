###### datetime:2022/09/24 15:52

###### author:nzb

# 新冠病毒疫情跟踪器API

- `router.py`

```python
from typing import List

import requests
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from pydantic import HttpUrl

from sqlalchemy.orm import Session

from .database import Base, engine, SessionLocal
from . import curd, schemas, models

application = APIRouter()

templates = Jinja2Templates(directory="./coronavirus/templates")

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@application.post("/create_city", response_model=schemas.ReadCity)
async def create_city(city: schemas.CreateCity, db: Session = Depends(get_db)):
    db_city = curd.get_city_by_name(db, name=city.province)
    if db_city:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="City already registered")
    return curd.create_city(db, city)


@application.get("/get_city/{city}", response_model=schemas.ReadCity)
async def get_city(city: str, db: Session = Depends(get_db)):
    db_city = curd.get_city_by_name(db, name=city)
    if db_city is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="City not found")
    return db_city


@application.get("/get_cities", response_model=List[schemas.ReadCity])
async def get_cities(offset: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    cities = curd.get_cities(db, offset=offset, limit=limit)
    return cities


@application.post("/create_data", response_model=schemas.ReadData)
async def ceate_data_for_city(city: str, data: schemas.CreateData, db: Session = Depends(get_db)):
    db_city = curd.get_city_by_name(db, name=city)
    if db_city is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="City not found")
    data = curd.create_city_data(db=db, data=data, city_id=db_city.id)
    return data


@application.get("/get_data")
def get_data(city: str = None, offset: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    data = curd.get_data(db, city=city, offset=offset, limit=limit)
    return data


def bg_task(url: HttpUrl, db: Session):
    """这里注意一个坑，不要在后台任务的参数中db: Session = Depends(get_db)这样导入依赖"""
    city_data = requests.get(url=f"{url}?source=jhu&country_code=CN&timelines=false")
    if city_data.status_code == 200:
        db.query(models.City).delete()  # 同步数据前先清空原有的数据
        for loc in city_data.json().get("locations", []):
            city = {
                "province": loc["province"],
                "country": loc["country"],
                "country_code": "CN",
                "country_population": loc["country_population"]
            }
            curd.create_city(db, schemas.CreateCity(**city))

    coronavirus_data = requests.get(url=f"{url}?source=jhu&country_code=CN&timelines=true")
    if coronavirus_data.status_code == 200:
        db.query(models.Data).delete()
        for city in coronavirus_data.json().get("locations", []):
            db_city = curd.get_city_by_name(db, name=city.get("province", ""))
            for date, confirmed in city["timelines"]['confirmed']['timeline'].items():
                data = {
                    "date": date.split("T")[0],  # 把'2020-12-31T00:00:00Z' 变成 ‘2020-12-31’
                    "confirmed": confirmed,
                    "deaths": city["timelines"]["deaths"]["timeline"][date],
                    "recovered": 0  # 每个城市每天有多少人痊愈，这种数据没有
                }
                # 这个city_id是city表中的主键ID，不是coronavirus_data数据里的ID
                curd.create_city_data(db, schemas.CreateData(**data), city_id=db_city.id)


@application.get("/sync_coronavirus_data/jhu")
async def sync_coronavirus_data(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """从Johns Hopkins University同步COVID-19数据"""
    background_tasks.add_task(bg_task, "https://coronavirus-tracker-api.herokuapp.com/v2/locations", db)
    return {"message": "正在后台同步数据..."}


@application.get("/")
async def coronavirus(request: Request, city: str = None, offset: int = 0, limit: int = 100,
                      db: Session = Depends(get_db)):
    data = curd.get_data(db, city=city, offset=offset, limit=limit)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "data": data,
        "sync_data_url": "/coronavirus/sync_coronavirus_data/jhu"
    })

```

- `database.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./coronavirus.sqlite3"
# SQLALCHEMY_DATABASE_URL = "postgresql://username:password@host:port/database_name"  # MySQL或PostgreSQL的连接方法

engine = create_engine(
    # echo=True表示引擎将用repr()函数记录所有语句及其参数列表到日志
    # 由于SQLAlchemy是多线程，指定check_same_thread=False来让建立的对象任意线程都可使用。这个参数只在用SQLite数据库时设置
    SQLALCHEMY_DATABASE_URL, encoding='utf-8', echo=True, connect_args={"check_same_thread": False}
)

# 在SQLAlchemy中，CRUD都是通过会话(session)进行的，所以我们必须要先创建会话，每一个SessionLocal实例就是一个数据库session
# flush()是指发送数据库语句到数据库，但数据库不一定执行写入磁盘；commit()是指提交事务，将变更保存到数据库文件
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=True)

# 创建基本映射类
Base = declarative_base(bind=engine, name="Base")

```

- `curd.py`

```python
from sqlalchemy.orm import Session

from . import models, schemas


def get_city(db: Session, city_id: int):
    return db.query(models.City).filter(models.City.id == city_id).first()


def get_city_by_name(db: Session, name: str):
    return db.query(models.City).filter(models.City.province == name).first()


def get_cities(db: Session, offset: int = 0, limit: int = 100):
    return db.query(models.City).offset(offset).limit(limit).all()


def create_city(db: Session, city: schemas.CreateCity):
    db_city = models.City(**city.dict())
    db.add(db_city)
    db.commit()
    db.refresh(db_city)
    return db_city


def get_data(db: Session, city: str = None, offset: int = 0, limit: int = 100):
    if city:
        return db.query(models.Data).filter(
            models.Data.city.has(province=city)).all()  # 外键关联查询，这里不是像Django ORM那样Data.city.province
    return db.query(models.Data).offset(offset).limit(limit).all()


def create_city_data(db: Session, data: schemas.CreateData, city_id: int):
    db_data = models.Data(**data.dict(), city_id=city_id)
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

```

- `models.py`

```python
from sqlalchemy import Column, String, Integer, BigInteger, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from .database import Base


class City(Base):
    __tablename__ = "city"  # 数据库的表名
    id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    province = Column(String(100), unique=True, nullable=False, comment="省/直辖市")
    country = Column(String(100), nullable=False, comment="国家")
    country_code = Column(String(100), nullable=True, comment="国家代码")
    country_population = Column(BigInteger, nullable=False, comment="国家人口")
    data = relationship('Data', back_populates="city")  # 'Data'是关联的类名；back_populates来指定反向访问的属性名称

    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")

    __mapper_args__ = {"order_by": country_code}  # 默认是正序，倒序加上.desc()方法

    def __repr__(self):
        return f"{self.country}_{self.province}"


class Data(Base):
    __tablename__ = "data"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    city_id = Column(Integer, ForeignKey('city.id'), comment="所属省/直辖市")  # ForeignKey里的字符串格式不是类名.属性名，而是表名.字段名
    date = Column(Date, nullable=False, comment="数据时间")
    confirmed = Column(BigInteger, default=0, nullable=False, comment="确诊数量")
    deaths = Column(BigInteger, default=0, nullable=False, comment="死亡数量")
    recovered = Column(BigInteger, default=0, nullable=False, comment="痊愈数量")
    city = relationship("City", back_populates='data')  # 'City'是关联的类名；back_populates来指定反向访问的属性名称

    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")

    __mapper_args__ = {"order_by": date.desc()}  # 按日期降序排列

    def __repr__(self):
        return f"{repr(self.date)}：确诊{self.confirmed}例"


""" 附上三个SQLAlchemy教程
SQLAlchemy的基本操作大全 
    http://www.taodudu.cc/news/show-175725.html
    
Python3+SQLAlchemy+Sqlite3实现ORM教程 
    https://www.cnblogs.com/jiangxiaobo/p/12350561.html
    
SQLAlchemy基础知识 Autoflush和Autocommit
    https://zhuanlan.zhihu.com/p/48994990
"""

```

- `schemas.py`

```python
from datetime import date as date_
from datetime import datetime

from pydantic import BaseModel


class CreateData(BaseModel):
    date: date_
    confirmed: int = 0
    deaths: int = 0
    recovered: int = 0


class CreateCity(BaseModel):
    province: str
    country: str
    country_code: str
    country_population: int


class ReadData(CreateData):
    id: int
    city_id: int
    updated_at: datetime
    created_at: datetime

    class Config:
        orm_mode = True


class ReadCity(CreateCity):
    id: int
    updated_at: datetime
    created_at: datetime

    class Config:
        orm_mode = True

```

- `home.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>新冠病毒疫情跟踪器</title>
    <link rel="stylesheet" href="{ { url_for('static', path='/semantic.min.css') } }">
    <script src="{ { url_for('static', path='/jquery-3.5.1/jquery-3.5.1.min.js') } }"></script>
    <script src="{ { url_for('static', path='/semantic.min.js') } }"></script>
    <script>
        $(document).ready(function () {
            $("#filter").click(function () {
                const city = $("#city").val();
                window.location.href = "http://" + window.location.host + "/coronavirus?city=" + city;
            });
            $("#sync").click(function () {
                $.get("{ { sync_data_url } }", function (result) {
                    alert("Message: " + result.message);
                });
            });
        });
    </script>
</head>

<body>
<div class="ui container">
    <h2></h2>
    <h1 style="text-align: center">新冠病毒疫情跟踪器</h1>
    <h2></h2>

    <button id="filter" style="float: left" type="submit" class="ui button alert-secondary">过滤</button>

    <div class="ui input">
        <label for="city"></label><input id="city" type="text" placeholder="城市" value="">
    </div>

    <button id="sync" style="float: right" type="submit" class="ui button primary">同步数据</button>

    <table class="ui celled table">
        <thead>
        <tr>
            <th>城市</th>
            <th>日期</th>
            <th>累计确诊数</th>
            <th>累计死亡数</th>
            <th>累计痊愈数</th>
            <th>更新时间</th>
        </tr>
        </thead>
        <tbody>
        {% for d in data %}
        <tr>
            <td>{ { d.city.province } }</td>
            <td>{ { d.date } }</td>
            <td>{ { d.confirmed } }</td>
            <td>{ { d.deaths } }</td>
            <td>{ { d.recovered } }</td>
            <td>{ { d.updated_at } }</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</div>
</body>
</html>
```