###### datetime:2022/09/24 15:44

###### author:nzb

# pydantic 基础使用

```python

from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, ValidationError, constr
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base

"""
Data validation and settings management using python type annotations.
使用Python的类型注解来进行数据校验和settings管理
pydantic enforces type hints at runtime, and provides user friendly errors when data is invalid.
Pydantic可以在代码运行时提供类型提示，数据校验失败时提供友好的错误提示
Define how data should be in pure, canonical python; validate it with pydantic.
定义数据应该如何在纯规范的Python代码中保存，并用Pydantic验证它
"""


class User(BaseModel):
    id: int  # 必填类型
    name: str = "Jack"  # 有默认值，选填字段
    signup_ts: Optional[datetime] = None  # 选填字段
    friends: List[int] = []  # 列表中元素是 int 类型或者可以直接转换成 int 类型


external_data = {
    "id": "123",
    "signup_ts": "2022-07-06 10:45",
    "friends": [1, 2, "3"]  # "3" 是可以 int("3")的
}
print("\033[31m1. --- Pydantic的基本用法。Pycharm可以安装Pydantic插件 ---\033[0m")
user = User(**external_data)
print(user.id, user.friends)
print(user.signup_ts)
print(user.dict())

print("\033[31m2. --- 校验失败处理 ---\033[0m")
try:
    User(id=1, signup_ts=datetime.today(), friends=[1, 2, 'not number'])
except ValidationError as e:
    print(e.json())

print("\033[31m3. --- 模型类的的属性和方法 ---\033[0m")
print(user.dict())
print(user.json())
print(user.copy())  # 浅拷贝
print(User.parse_obj(external_data))
print(User.parse_raw('{"id": "123", "signup_ts": "2020-12-22 12:22", "friends": [1, 2, "3"]}'))

path = Path("pydantic_tutorial.json")
path.write_text('{"id": "123", "signup_ts": "2020-12-22 12:22", "friends": [1, 2, "3"]}')
print(User.parse_file(path))

print(user.schema())
print(user.schema_json())

user_data = {"id": "error", "signup_ts": "2020-12-22 12:22", "friends": [1, 2, 3]}  # id是字符串 是错误的
print(User.construct(**user_data))  # 不检验数据直接创建模型类，不建议在construct方法中传入未经验证的数据
# name='Jack' signup_ts='2020-12-22 12:22' friends=[1, 2, 3] id='error'

print(User.__fields__.keys())  # 定义模型类的时候，所有字段都注明类型，字段顺序就不会乱

print("\033[31m4. --- 递归模型 ---\033[0m")


class Sound(BaseModel):
    sound: str


class Dog(BaseModel):
    birthday: date
    weight: float = Optional[None]
    sound: List[Sound]  # 不同的狗有不同的叫声。递归模型（Recursive Models）就是指一个嵌套一个


dogs = Dog(birthday=date.today(), weight=6.66, sound=[{"sound": "wang wang ~"}, {"sound": "ying ying ~"}])
print(dogs.dict())

print("\033[31m5. --- ORM模型：从类实例创建符合ORM对象的模型  ---\033[0m")

Base = declarative_base()


class ComanyOrm(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, nullable=True)
    public_key = Column(String(20), index=True, nullable=True, unique=True)
    name = Column(String(64), unique=True)
    domains = Column(ARRAY(String(255)))


class ComanyModel(BaseModel):
    id: int
    public_key: constr(max_length=20)
    name: constr(max_length=64)
    domains: List[constr(max_length=255)]

    class Config:
        orm_mode = True


company_orm = ComanyOrm(
    id=123,
    public_key="foobar",
    name="Testing",
    domains=["demo.com", "example.com"]
)

print(ComanyModel.from_orm(company_orm))

print("\033[31m6. --- Pydantic支撑的字段类型,官方文档：https://pydantic-docs.helpmanual.io/usage/types/  ---\033[0m")

```

```text
1. --- Pydantic的基本用法。Pycharm可以安装Pydantic插件 ---
123 [1, 2, 3]
2022-07-06 10:45:00
{'id': 123, 'name': 'Jack', 'signup_ts': datetime.datetime(2022, 7, 6, 10, 45), 'friends': [1, 2, 3]}
2. --- 校验失败处理 ---
[
  {
    "loc": [
      "friends",
      2
    ],
    "msg": "value is not a valid integer",
    "type": "type_error.integer"
  }
]
3. --- 模型类的的属性和方法 ---
{'id': 123, 'name': 'Jack', 'signup_ts': datetime.datetime(2022, 7, 6, 10, 45), 'friends': [1, 2, 3]}
{"id": 123, "name": "Jack", "signup_ts": "2022-07-06T10:45:00", "friends": [1, 2, 3]}
id=123 name='Jack' signup_ts=datetime.datetime(2022, 7, 6, 10, 45) friends=[1, 2, 3]
id=123 name='Jack' signup_ts=datetime.datetime(2022, 7, 6, 10, 45) friends=[1, 2, 3]
id=123 name='Jack' signup_ts=datetime.datetime(2020, 12, 22, 12, 22) friends=[1, 2, 3]
id=123 name='Jack' signup_ts=datetime.datetime(2020, 12, 22, 12, 22) friends=[1, 2, 3]
{'title': 'User', 'type': 'object', 'properties': {'id': {'title': 'Id', 'type': 'integer'}, 'name': {'title': 'Name', 'default': 'Jack', 'type': 'string'}, 'signup_ts': {'title': 'Signup Ts', 'type': 'string', 'format': 'date-time'}, 'friends': {'title': 'Friends', 'default': [], 'type': 'array', 'items': {'type': 'integer'}}}, 'required': ['id']}
{"title": "User", "type": "object", "properties": {"id": {"title": "Id", "type": "integer"}, "name": {"title": "Name", "default": "Jack", "type": "string"}, "signup_ts": {"title": "Signup Ts", "type": "string", "format": "date-time"}, "friends": {"title": "Friends", "default": [], "type": "array", "items": {"type": "integer"}}}, "required": ["id"]}
name='Jack' signup_ts='2020-12-22 12:22' friends=[1, 2, 3] id='error'
dict_keys(['id', 'name', 'signup_ts', 'friends'])
4. --- 递归模型 ---
{'birthday': datetime.date(2022, 9, 24), 'sound': [{'sound': 'wang wang ~'}, {'sound': 'ying ying ~'}]}
5. --- ORM模型：从类实例创建符合ORM对象的模型  ---
id=123 public_key='foobar' name='Testing' domains=['demo.com', 'example.com']
6. --- Pydantic支撑的字段类型,官方文档：https://pydantic-docs.helpmanual.io/usage/types/  ---
```