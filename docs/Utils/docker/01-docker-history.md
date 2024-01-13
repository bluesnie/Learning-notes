###### datetime:2023/03/31 17:48

###### author:nzb

# 构建过程

- 1、构建过程
    - 从基础镜像运行一个容器
    - 执行一条指令，对容器做出修改
    - 执行类似`docker commit`的操作，提交一个新的镜像层
    - 在基于刚提交的镜像运行一个新容器
    - 执行Dockerfile中的下一条指令，直至所有指令执行完毕

      > ps:构建中会删除中间层容器，而不会删除中间层镜像，所以可以使用中间层镜像进行调试，查找错误

- 2、镜像缓存
    - 构建缓存:
      构建一次后再构建就会使用构建缓存
    - 不使用缓存
        - 使用--no-cache选项
        - 或
        - `ENV REFRESH_DATE 2019-4-7`

3、查看镜像构建的过程
- `docker history [image]`
    - `--no-trunc`: 不截断输出完整信息

- 只显示构建命令
    - `docker history --format { {.CreatedBy} } --no-trunc=true e01eb2e99ca6 |sed "s?/bin/sh\ -c\ \#(nop)\ ??g"|sed "s?/bin/sh\ -c?RUN?g" | tac`

    - 示例

        ```text
        ADD file:7dc8819fd3d4b84ad19fb836e5bfda64a5ffefc371166f70d4d41dff6b22d450 in / 
        RUN [ -z "$(apt-get indextargets)" ]
        RUN set -xe   && echo '#!/bin/sh' > /usr/sbin/policy-rc.d  && echo 'exit 101' >> /usr/sbin/policy-rc.d  && chmod +x /usr/sbin/policy-rc.d   && dpkg-divert --local --rename --add /sbin/initctl  && cp -a /usr/sbin/policy-rc.d /sbin/initctl  && sed -i 's/^exit.*/exit 0/' /sbin/initctl   && echo 'force-unsafe-io' > /etc/dpkg/dpkg.cfg.d/docker-apt-speedup   && echo 'DPkg::Post-Invoke { "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true"; };' > /etc/apt/apt.conf.d/docker-clean  && echo 'APT::Update::Post-Invoke { "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true"; };' >> /etc/apt/apt.conf.d/docker-clean  && echo 'Dir::Cache::pkgcache ""; Dir::Cache::srcpkgcache "";' >> /etc/apt/apt.conf.d/docker-clean   && echo 'Acquire::Languages "none";' > /etc/apt/apt.conf.d/docker-no-languages   && echo 'Acquire::GzipIndexes "true"; Acquire::CompressionTypes::Order:: "gz";' > /etc/apt/apt.conf.d/docker-gzip-indexes   && echo 'Apt::AutoRemove::SuggestsImportant "false";' > /etc/apt/apt.conf.d/docker-autoremove-suggests
        RUN mkdir -p /run/systemd && echo 'docker' > /run/systemd/container
         CMD ["/bin/bash"]
        RUN echo 'Etc/UTC' > /etc/timezone &&     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime &&     apt-get update &&     apt-get install -q -y --no-install-recommends tzdata &&     rm -rf /var/lib/apt/lists/*
        RUN apt-get update && apt-get install -q -y --no-install-recommends     dirmngr     gnupg2     && rm -rf /var/lib/apt/lists/*
        RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
        RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list
         ENV LANG=C.UTF-8
         ENV LC_ALL=C.UTF-8
         ENV ROS_DISTRO=melodic
        RUN apt-get update && apt-get install -y --no-install-recommends     ros-melodic-ros-core=1.4.1-0*     && rm -rf /var/lib/apt/lists/*
        COPY file:cbbaa0f5d6a276512315f5b4d7347e94a120cefbda9058ebb0d678847ff4837f in / 
         ENTRYPOINT ["/ros_entrypoint.sh"]
         CMD ["bash"]
        RUN apt-get update && apt-get install --no-install-recommends -y     build-essential     python-rosdep     python-rosinstall     python-vcstools     && rm -rf /var/lib/apt/lists/*
        RUN rosdep init &&   rosdep update --rosdistro $ROS_DISTRO
        RUN apt-get update && apt-get install -y --no-install-recommends     ros-melodic-ros-base=1.4.1-0*     && rm -rf /var/lib/apt/lists/*
        RUN apt-get update && apt-get install -y --no-install-recommends     ros-melodic-robot=1.4.1-0*     && rm -rf /var/lib/apt/lists/*
        RUN sed -i 's/ports.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
        RUN apt-get update
        RUN apt-get install -y openssh-server
        RUN echo -e 'y\n'|ssh-keygen -q -t rsa -N "" -f ~/.ssh/id_rsa
        RUN apt-get remove -y openssh-server
        bash
        RUN apt install -y python-pip
        RUN pip install serial -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install pyserial -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install flask -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install flask_cors -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install Twisted  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install flask_sockets -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install pyjson -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install protobuf -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install requests -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN pip install zmq -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
        RUN apt install -y iftop
        RUN mkdir -p /config && mkdir -p /logs && mkdir -p /data && mkdir -p /walle
        WORKDIR /upper_computer
        RUN cd /upper_computer &&git pull &&git checkout dev_qys_lansi_new &&rm -rf install &&/bin/bash -c 'source "/opt/ros/melodic/setup.bash" &&catkin_make install' &&cp -r /upper_computer/src/upper_computer_ui/script/upper_computer_ui/dist           /upper_computer/install/lib/python2.7/dist-packages/upper_computer_ui/
         ENTRYPOINT ["/upper_computer/start.sh"]
         CMD ["/bin/bash"]
         CMD ["/bin/bash"]
        bash
        RUN cd /upper_computer &&git pull &&git checkout prd_master_alpha &&git pull &&rm -rf install &&/bin/bash -c 'source "/opt/ros/melodic/setup.bash" &&\catkin_make install'
         ENTRYPOINT ["/upper_computer/start.sh"]
         CMD ["/bin/bash"]
         CMD ["/bin/bash"]
        bash
        bash
        bash
         ENV DIRPATH=/tmp/py39
        WORKDIR /tmp/py39
        COPY file:a37b26f8d2f91243c4ffc8eaf134fba1ff8f257488060568c5b9f6ff705ec716 in /tmp/py39 
        COPY file:a6cac2b37ef882b75b365dc79d169031648a6f4f1322be042b9e4edf784b7e37 in /tmp/py39 
        RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
        RUN apt-get clean
        RUN apt-get update && apt-get -y upgrade
        RUN apt-get install -y build-essential python-dev python-setuptools python-pip python-smbus build-essential libncursesw5-dev libgdbm-dev libc6-dev zlib1g-dev libsqlite3-dev tk-dev libssl-dev openssl libffi-dev
        RUN tar -zxvf Python-3.9.0b4.tgz
        WORKDIR /tmp/py39/Python-3.9.0b4
        RUN ./configure --prefix=/usr/local/python39 --with-ssl --enable-optimizations
        RUN make
        RUN make install
        RUN ln -s /usr/local/python39/bin/python3.9 /usr/bin/python3.9 && ln -s /usr/local/python39/bin/pip3.9 /usr/bin/pip3.9
        WORKDIR /tmp/py39
        RUN pip3.9 install -r requirements_py39.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        RUN rm -rf $DIRPATH
        WORKDIR /upper_computer
        RUN cd /upper_computer &&git pull &&git checkout prd_master_alpha &&git pull &&/bin/bash -c 'source "/opt/ros/melodic/setup.bash" &&\catkin_make install'
         ENTRYPOINT ["/upper_computer/start.sh"]
         CMD ["/bin/bash"]
        --name test_082501
        COPY file:80e64585a1026126a9ce85c15b4f1bfaf23abe894769170ce033ab4a4c768ed9 in /upper_computer/ 
        RUN cd /upper_computer &&pip3.9 install zmq numpy serial pyserial protobuf==3.20.1 -i https://pypi.douban.com/simple &&git pull &&git checkout prd_master_alpha &&git pull &&rm -rf install &&/bin/bash -c 'source "/opt/ros/melodic/setup.bash" &&\catkin_make install'
         ENTRYPOINT ["/upper_computer/start.sh"]
         CMD ["/bin/bash"]
        RUN pip install openpyxl==2.6.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
        RUN pip3.9 install openpyxl==2.6.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
        RUN cd /upper_computer &&rm -f version &&git checkout prd_master_alpha &&git pull &&git checkout phoenix_master &&git pull &&rm -rf install/ build/ devel/ &&/bin/bash -c 'source "/opt/ros/melodic/setup.bash" &&\catkin_make install'
         ENTRYPOINT ["/upper_computer/start.sh"]
         CMD ["/bin/bash"]
        RUN pip3.9 install modbus_tk xlrd -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
        RUN cd /upper_computer &&rm -f version &&git checkout prd_master_alpha &&git pull &&git checkout phoenix_master &&git pull &&rm -rf install/ build/ devel/ &&/bin/bash -c 'source "/opt/ros/melodic/setup.bash" &&\catkin_make install'
         ENTRYPOINT ["/upper_computer/start.sh"]
         CMD ["/bin/bash"]
        ```




















