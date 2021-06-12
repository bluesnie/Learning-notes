# 建立python3.6环境
FROM python:3.6

# 作者
MAINTAINER NZB

# 设置Python环境
ENV PYTHONUNBUFFERED 1

COPY pip.conf /root/.pip/pip.conf

# 创建django_demo文件夹
RUN mkdir -p /opt/python_project/django_demo

# 将django_demo文件夹设为工作目录
WORKDIR /opt/python_project/django_demo

# 将当前目录加入到工作目录中（.  表示当前目录）
ADD . /opt/python_project/django_demo

# 更新pip版本
RUN /usr/local/bin/python -m pip install --upgrade pip

# 利用pip安装依赖
RUN pip install -r requirements.txt

# 去除Windows系统编辑文件中多余的\r回车空格
RUN sed -i 's/\r//' ./start.sh

# 给start.sh可执行权限
RUN chmod  +x ./start.sh