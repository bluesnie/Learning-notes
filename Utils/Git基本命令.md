# Git常用命令

## 1、帮助信息

        git help           显示常用的git 和使用简短说明
        git help -a       显示所有的命令
        git help -g       查看使用手册
        git help 命令 / git 命令 help     查看某命令的使用说明, F键下一页，B键上一页，Q退出

## 2、git 配置（全局配置）
    
        所有的配置都会保存到当前用户目录下的: .gitconfig 文件中
            git config --global user.name '名称'               配置用户名
            git config --global user.email '邮箱名'            配置邮箱
            git config --list                                 查看配置信息    
            git config --unset --global user.name '名称'       重置信息
            git config --global corlor.ui true

## 3、初始化项目

        git init        初始化项目
    
## 4、查看状态
    
        git status
            状态：
                untracked:未跟踪的文件
                modified: 修改后未添加提交的文件
        
## 5、添加文件
    
        git add .或具体文件                      添加当前文件夹的文件或具体文件

## 6、提交文件
    
        git commit -m '提交信息'                提交
        git commit -am '提交信息'               添加提交

## 7、查看提交日志

        git log --oneline --decorate --all --10 --graph --author='作者' --index='文件名' --before='2019-3-1/1 week' 
            --oneline:一行显示提交日志
            --decorate:显示详细
            --all:显示在所有分支上的提交
            --10:显示数量
            --graph:显示分支信息
            --author:指定作者
            --grep:搜索某文件
            --before:某时间之前
        
    
## 8、查看文件修改前和修改后的区别

        git diff 文件名                        查看文件修改的区别，不指明文件则所以修改文件的区别

## 9、Git跟踪rename文件/移动文件

        git mv 原文件名 新文件名                重命名/移动文件夹或文件名
        git add .
        git commit -m '信息'
    
## 10、删除文件

        git rm 文件名1 文件名2 。。。    
        git rm -r 文件夹名                      递归删除  
    
## 11、恢复文件

        git checkout HEAD^ -- 需要恢复的文件名
            HEAD:最近的一次提交
            HEAD^:最近的一次提交的上一次提交
            HEAD^^:最近的一次提交的上两次提交
            HEAD^^...:最近的一次提交的上n次提交
            --:当前分支

## 12、恢复提交

        git revert 提交号

## 13、重置提交指针

        git reset 选项 提交号
            --soft:软重置，不会影响工作区和暂存区的东西
            --hard:工作区和暂存区直接重置到指定的状态
            --mixed: 默认，会把暂存区重置到指定的状态，并把指针指到当前位置
        
## 14、查看/创建/切换分支

        git branch -a               查看分支
        git branch -r               查看远程分支    
        git branch 分支名            创建分支  
        git checkout 分支名          切换分支

## 15、查看两个分支之间的区别

        git diff master..branch1 文件名   查看两个分支（文件）之间的区别，a表示两点左边的分支，b表示右边的分支

## 16、合并分支

        git checkout master
        git merge 分支名    
    
## 17、解决合并冲突

        手动解决冲突
        Git用<<<<<<<，=======，>>>>>>>标记出不同分支的内容

## 18、重命名/删除分支

        git branch -m 原分支名 新分支名
        git branch -d 分支名

## 19、保存修改进度

        git stash save '描述信息'
        git stash list                  查看工作进度信息
        git stash show -p 工作进度代号    查看工作进度和现在的区别
        git apply 工作进度代号            恢复工作进度
        git shash drop 工作进度代号       删除工作进度
        git apply pop 工作进度代号        恢复工作进度同时删除
    
## 20、添加别名

        git config --global alias.co(别名) checkout(命令)
        或
        编辑当前用户文件夹下的.bash_profile文件
            alias gco='git checkout'
        保存退出
        source ~/.bash_profile或重启终端

## 21、全局忽略跟踪文件

        git config --global core.excludesfile ~/.gitignore_global
        告诉git全局范围中忽略的文件包含在.gitignore_global文件中
        编辑.gitignore_global需要忽略的文件
    
## 22、项目级忽略文件

        在项目根目录下创建.gitignore文件
        在.gitignore文件下添加忽略文件
    
## 23、创建远程版本库
    
        git remote add origin 远程版本库url地址
        git remote -v                               查看远程库信息
        git remote rm                               移除远程库
    
## 24、推送版本库

        git push [-u] origin 分支名
            -u:跟踪远程分支的变化

## 25、克隆版本库到本地

        git clone 远程版本库url地址 目录名            克隆到指定目录下

## 26、更新本地版本库

        git fetch                                   拉取版本库
        git merge origin/master                     合并
        或
        git pull = git fetch + git merge
        第一种比较安全

## 27、基于版本库开发自己的版本库，fork到自己账户然后克隆到本地

        git fork 远程版本库url地址
        
## 28、添加pull request

        git pull request
    
## 29、添加贡献者

        GitHub中的setting中的collaborator添加贡献者
    
    
                                
