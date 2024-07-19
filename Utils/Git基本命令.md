# Git常用命令

## 帮助信息

        git help           显示常用的git 和使用简短说明
        git help -a       显示所有的命令
        git help -g       查看使用手册
        git help 命令 / git 命令 help     查看某命令的使用说明, F键下一页，B键上一页，Q退出

## git 配置（全局配置）

        所有的配置都会保存到当前用户目录下的: .gitconfig 文件中
            git config --global user.name '名称'               配置用户名
            git config --global user.email '邮箱名'            配置邮箱
            git config --list                                 查看配置信息    
            git config --unset --global user.name '名称'       重置信息
            git config --global corlor.ui true

## 初始化项目

        git init        初始化项目

## 查看状态

        git status
            状态：
                untracked:未跟踪的文件
                modified: 修改后未添加提交的文件

## 添加文件

        git add .或具体文件                      添加当前文件夹的文件或具体文件

## 提交文件

        git commit -m '提交信息'                提交
        git commit -am '提交信息'               添加提交

## 查看提交日志

        git log --oneline --decorate --all -10 --graph --author='作者' --index='文件名' --before='2019-3-1/1 week' 
            --oneline:一行显示提交日志
            --decorate:显示详细
            --all:显示在所有分支上的提交
            -10:显示数量
            --graph:显示分支信息
            --author:指定作者
            --grep:搜索某文件
            --before:某时间之前

## 查看文件修改前和修改后的区别

        git diff 文件名                        查看文件修改的区别，不指明文件则所以修改文件的区别

## Git跟踪rename文件/移动文件

        git mv 原文件名 新文件名                重命名/移动文件夹或文件名
        git add .
        git commit -m '信息'

## 删除文件

        git rm 文件名1 文件名2 。。。    
        git rm -r 文件夹名                      递归删除  

## 恢复文件

        git checkout HEAD^ -- 需要恢复的文件名
            HEAD:最近的一次提交
            HEAD^:最近的一次提交的上一次提交
            HEAD^^:最近的一次提交的上两次提交
            HEAD^^...:最近的一次提交的上n次提交
            --:当前分支

## 恢复提交

        git revert 提交号

## 重置提交指针

        git reset 选项 提交号
            --soft:软重置，不会影响工作区和暂存区的东西
            --hard:工作区和暂存区直接重置到指定的状态
            --mixed: 默认，会把暂存区重置到指定的状态，并把指针指到当前位置
        
        git status 先看一下add 中的文件 
        git reset HEAD 如果后面什么都不跟的话 就是上一次add 里面的全部撤销了 
        - HEAD^ 表示上一个版本，即上一次的commit，也可以写成HEAD~1
        - 如果进行两次的commit，想要都撤回，可以使用HEAD~2
        git reset HEAD XXX/XXX/XXX.java 就是对某个文件进行撤销了

## 查看/创建/切换分支

        git branch -a               查看分支
        git branch -r               查看远程分支    
        git branch 分支名            创建分支  
        git checkout 分支名          切换分支

## 查看两个分支之间的区别

        git diff master..branch1 文件名   查看两个分支（文件）之间的区别，a表示两点左边的分支，b表示右边的分支

## 合并分支

        git checkout master
        git merge 分支名    

## 解决合并冲突

        手动解决冲突
        Git用<<<<<<<，=======，>>>>>>>标记出不同分支的内容

## 重命名/删除分支

        git branch -m 原分支名 新分支名
        git branch -d 分支名

## 保存修改进度

        git stash save '描述信息'
        git stash list                  查看工作进度信息
        git stash show -p 工作进度代号    查看工作进度和现在的区别
        git apply 工作进度代号            恢复工作进度
        git shash drop 工作进度代号       删除工作进度
        git apply pop 工作进度代号        恢复工作进度同时删除

## 添加别名

        git config --global alias.co(别名) checkout(命令)
        或
        编辑当前用户文件夹下的.bash_profile文件
            alias gco='git checkout'
        保存退出
        source ~/.bash_profile或重启终端

## 全局忽略跟踪文件

        git config --global core.excludesfile ~/.gitignore_global
        告诉git全局范围中忽略的文件包含在.gitignore_global文件中
        编辑.gitignore_global需要忽略的文件

## 项目级忽略文件

        在项目根目录下创建.gitignore文件
        在.gitignore文件下添加忽略文件
        
        如果你不想推什么文件到git 可以运行这个命令：
        git update-index --assume-unchanged xxx/xxx.py  

## 忽略已被跟踪的文件

- 忽略规则只针对还没有被git跟踪的文件及文件夹有效。若需要忽略规则对已被跟踪的文件及文件夹有效，则需要取消对文件或文件夹的跟踪

    - `git rm -r --cached <dir>`：取消对文件夹及文件夹下的所有子文件夹、文件的跟踪，文件夹及文件夹下的所有子文件夹、文件的状态将从跟踪状态变为未跟踪状态

    - `git rm --cached <file>`：取消对文件的跟踪，文件的跟踪状态将变为未跟踪状态

- 取消对文件或文件夹的跟踪之后，`.gitignore`文件中的忽略规则将会对取消了跟踪状态的文件或文件夹生效

## 创建远程版本库

        git remote add origin 远程版本库url地址
        git remote -v                               查看远程库信息
        git remote rm                               移除远程库

## 推送版本库

        git push [-u] origin 分支名
            -u:跟踪远程分支的变化

## 修改远程仓库地址

        1.修改命令
            git remote origin set-url [url]
        
        2.先删后加
            git remote rm origin
            git remote add origin [url]
        
        3.直接修改config文件

## 克隆版本库到本地

        git clone 远程版本库url地址 目录名            克隆到指定目录下

## 更新本地版本库

        git fetch                                   拉取版本库
        git merge origin/master                     合并
        或
        git pull = git fetch + git merge
        第一种比较安全

## 基于版本库开发自己的版本库，fork到自己账户然后克隆到本地

        git fork 远程版本库url地址

## 添加pull request

        git pull request

## 添加贡献者

        GitHub中的setting中的collaborator添加贡献者

## 详情图

![image](./docker/img/git.png)

## git submodule常用命令

- 添加子模块

```shell
git submodule add <repository> <path>
```
> `<path>` 是子模块在当前仓库中的相对路径

- 初始化子模块

```shell
git submodule update --init --recursive
```

- 更新子模块

```shell
git submodule update --remote <path>
```

- 切换子模块分支

```shell
cd <path>
git checkout <branch>
cd ..
git add <path>
git commit -m "Update submodule"
```

- 删除子模块

```shell
git submodule deinit -f <path/name>  # path 可以在.gitmodule里面查看,使用 -f 选项来放弃子模块目录中的任何本地更改。 如果没有这个选项，子模块中如果有任何未提交的更改，命令可能会失败。
rm -rf .git/modules/<path/name>
git config -f .gitmodules --remove-section submodule.<path/name>
$ git add .gitmodules
git rm --cached <path/name>
git add .
git commit -m 'rm submodule:<path/name>'
```

- 递归克隆包含子项目的仓库

```shell
git clone --recurse-submodules <repository>
```

### git submodule foreach 
- 切换全部子项目分支

```shell
git submodule foreach 'git checkout <branch_name>'
```

- 更新拉取所有子项目
```shell
git submodule foreach git pull
```

- 解决 git 在子模块中执行报错后直接退出，而是继续执行
```shell
git submodule foreach '其他命令 || :'
或
git submodule foreach '其他命令 || true'
```

- 选择跳过
```shell
 git submodule foreach 'case $name in a-Module|b-Module|c-Module) ;; *) git status ;; esac'  
 # 两个;;前都可以加命令
 git submodule foreach 'case $name in a-Module|b-Module|c-Module) echo "Processing submodule----------->: $name" ;; *) git status ;; esac'
```
这个命令是去批量执行，如果遇到 a-Module，b-Module，c-Module 中的任何一个，什么都不操作，其他的子模块中，执行 git status。
测试后，的确可以跳过 a-Module|b-Module|c-Module 这三个模块进行处理。

- 选择某些子项目操作
```shell
git submodule foreach 'case $name in a-Module|b-Module) git  branch;; esac'
```

## 多个github账号克隆

- [配置连接](https://blog.csdn.net/meng_feng12/article/details/131866917)
- `clone`使用(`submodule`不支持)： `git clone git@github.com:test/hello.git --config=core.sshCommand="ssh -i ~/.ssh/id_rsa"`

### 配置步骤

- 删除全局`github`账号
  - 查看：`git config --list`
  - 删除：`git config --global --unset user.name`
  - 删除：`git config --global --unset user.email`

- 进入或创建`.ssh`
  - 进入：`cd ~/.ssh`
  - 创建：`mkdir ~/.ssh`
- 生成`ssh`密钥
  - 生成：`ssh-keygen -t rsa -f ~/.ssh/id_rsa_user1 -C "yourmail1@xxx.com"`
  - 生成：`ssh-keygen -t rsa -f ~/.ssh/id_rsa_user2 -C "yourmail2@xxx.com"`
  
  然后回车后按照提示即可生成密钥，默认的文件名是`id_rsa_user1`。其中`id_rsa_user1`为私钥，`id_rsa_user1.pub`为公钥。为了方便区分不同的`git`账户，这里修改密钥文件名为：`id_rsa_user1`

- 将私钥添加到本地

`SSH`协议的原理就是在托管平台上使用公钥，在本地使用私钥，这样本地仓库就可以和远程仓库进行通信。在上一步已经生成密钥对，接下来需要把私钥添加到本地：

```shell
ssh-add ~/.ssh/id_rsa_user1 // 将私钥添加到本地
ssh-add ~/.ssh/id_rsa_user2 // 将私钥添加到本地
```
为了检验本地是否添加成功，可以使用 `ssh-add -l` 命令进行查看

- `git`托管账户绑定`ssh`

以`github`为例，其他的托管平台类似：
复制 `id_rsa_github.pub` 文件中的内容，然后打开`github`网站，右上角点击头像，然后找到 `Settings` 并点击，然后找到 `SSH keys and GPG keys` ，点击 `New SSH Key`，将复制的内容粘贴到输入框内，然后点击添加按钮即可。

- 配置账户密匙管理文件

由于添加了多个平台的密钥文件，所以需要对这些密钥进行管理。在 `.ssh` 目录下新建一个 `config` 文件（注意这个`config`文件并不是`.txt`等文件，而是一个不带任何后缀名的文件）：
执行命令：`touch ~/.ssh/config`

```text
# user1
Host user1.github.com // 网站的别名，随意取
HostName github.com // 托管网站的域名
User user1 // github上的用户名
IdentityFile ~/.ssh/id_rsa_user1 // 使用的密钥文件

# user2
Host user2.github.com // 网站的别名，随意取
HostName github.com
User user2
IdentityFile ~/.ssh/id_rsa_user2

```

- 测试连接

```shell
ssh -T git@github.com  // 使用托管平台域名(如果多个github账号，需要使用别名)

ssh -T git@user1.github.com     // 使用托管平台别名
ssh -T git@user2.github.com
#  Hi user1! You've successfully authenticated, but GitHub does not provide shell access.
```

- 管理git的用户名和邮箱

```shell
查看全局配置
git config --global user.name
git config --global user.email

查看本地配置(只能在git仓库中使用)
git config --local user.name
git config --local user.email
```

- 使用 `git` 重要
  - `git` 的使用一般是从其他仓库直接 `clone` 或本地新建，注意配置用户名和邮箱。
    - `clone` 到本地 原来写法 `git clone git@github.com:用户名/learngit.git`
    - 现在写法
    
```shell
git clone git@user1.github.com:user1/learngit.git
git clone git@user2.github.com:user2/learngit.git
``` 

- 如何提交代码

```shell
# push 到 github上去
git remote rm origin //清空原有的
git remote add origin git@user1.github.com:user1/test.git
# git remote add 是 Git 命令，用于添加一个新的远程仓库。


git push --set-upstream origin main
git push 
```
