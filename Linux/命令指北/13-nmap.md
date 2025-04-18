###### datetime:2025/04/18 11:15:00

###### author:nzb

# Linux命令之nmap命令

## 文章目录
- 一、目标指定
- 二、扫描类型
- 三、主机发现
- 四、端口扫描
- 五、服务和版本检测
- 六、操作系统检测
- 七、输出
- 八、定时和性能
- 九、防火墙规避技术
- 十、Nmap脚本引擎
- 十一、结果处理
- 十二、扫描IP范围

## 一、目标指定

```bash
# 扫描单个IP
nmap <target>
# 扫描多个指定IP
nmap <target1> <target2>
# 扫描IP范围
nmap <range>
# 扫描域名
nmap <domain>
# 使用CIDR表示法扫描
nmap <CIDR>
# 从文件中读取目标进行扫描
nmap -iL <file>
# 扫描指定数量的随机主机
nmap -iR <count>
# 排除特定主机
nmap --exclude <target>
```

## 二、扫描类型

```bash
# TCP SYN扫描（默认）
nmap -sS <target>
# TCP Connect扫描
nmap -sT <target>
# UDP扫描
nmap -sU <target>
# 隐蔽扫描
nmap -sF | -sX | -sN <target>
```

## 三、主机发现

```bash
# Ping扫描，识别存活主机
nmap -sn <target>
# 仅列出目标，不进行扫描
nmap -sL <target>
# 使用TCP ACK数据包进行发现
nmap -PA<ports> <target>
```

## 四、端口扫描

```bash
# 扫描指定端口
nmap -p <port> <target>
# 扫描端口范围
nmap -p <range> <target>
# 扫描所有端口
nmap -p- <target>
# 快速扫描前100个端口
nmap -F <target>
```

## 五、服务和版本检测

```bash
# 尝试识别服务和版本
nmap -sV <target>
# 更快但准确性较低的检测模式
nmap -sV --version-light <target>
# 启用所有强度级别的检测
nmap -sV --version-all <target>
```

## 六、操作系统检测

```bash
# 识别操作系统
nmap -O <target>
# 进行积极的操作系统猜测
nmap -O --osscan-guess <target>
```

## 七、输出

```bash
# 将普通输出保存到文件
nmap -oN <file> <target>
# 将XML输出保存到文件
nmap -oX <file> <target>
# 仅显示开放端口
nmap --open <target>
```

## 八、定时和性能

```bash
# 调整扫描速度（0-5）
nmap -T<0-5> <target>
# 限制每秒的探测数量
nmap --max-rate <rate> <target>
```

## 九、防火墙规避技术

```bash
# 调整扫描时间以避免检测
nmap -T2 -Pn <target>
# 数据包分片以规避防火墙
nmap -f <target>
# 设置自定义MTU大小进行数据包分片
nmap --mtu <size> <target>
# 随机化目标顺序以规避检测
nmap --randomize-hosts -iL <file>
# 使用欺骗IP地址混淆真实扫描器
nmap -D RND:<count>,ME <target>
# 伪装扫描源IP
nmap -S <ip> <target>
# 伪装MAC地址
nmap --spoof-mac <mac> <target>
# 伪装源端口
nmap --source-port <port> <target>
# 自定义TTL以绕过防火墙规则
nmap --ttl <value> <target>
# 通过代理进行扫描以保持匿名
nmap --proxies <proxylist> <target>
# 使用ICMP、SCTP或其他非常见协议
nmap -PE <target>
# 添加任意数据以混淆IDS
nmap --data-length <length> <target>
# 进行DNS查找而不是直接扫描
nmap -sL <target>
# 结合多种规避技术进行高级扫描
nmap -f -T2 -D RND:5 --spoof-mac 0 --source-port 443 <target>
```

## 十、Nmap脚本引擎

```bash
# DNS暴力枚举
nmap --script=dns-brute --script-args dns-brute.domain=<domain> <target>
# HTTP枚举
nmap --script=http-enum <target>
# SSH暴力破解
nmap -p 22 --script=ssh-brute --script-args userdb=<userfile>,passdb=<passfile> <target>
# SMB枚举共享
nmap -p 445 --script=smb-enum-shares --script-args smbuser=<user>,smbpass=<pass> <target>
# MySQL暴力破解
nmap -p 3306 --script=mysql-brute --script-args userdb=<userfile>,passdb=<passfile> <target>
# HTTP内容抓取
nmap -p 80 --script=http-grep --script-args http-grep.url=<subpage> <target>
# HTTP配置备份枚举
nmap -p 80 --script=http-config-backup <target>
# SMB枚举用户
nmap --script=smb-enum-users <target>
# HTTP WordPress枚举
nmap -p 80 --script=http-wordpress-enum <target>
# 穿过防火墙扫描
nmap --script=firewalk <target>
# MySQL空密码检测
nmap -p 3306 --script=mysql-empty-password <target>
# MySQL用户枚举
nmap -p 3306 --script=mysql-users --script-args mysqluser=<user>,mysqlpass=<pass> <target>
# SMB操作系统发现
nmap --script=smb-os-discovery <target>
# DNS区域传输
nmap --script=dns-zone-transfer --script-args dns-zone-transfer.domain=<domain> <target>
# FTP匿名登录检测
nmap --script=ftp-anon <target>
# SMTP用户枚举
nmap --script=smtp-enum-users --script-args smtp.domain=<domain> <target>
# 漏洞扫描
nmap --script=vulners --script-args mincvss=<value> <target>
# SNMP暴力破解
nmap --script=snmp-brute <target>
# HTTP漏洞扫描
nmap --script=http-vuln-* <target>
# SMB枚举共享
nmap --script=smb-enum-shares <target>
# HTTP标题检测
nmap -p 80,443 --script=http-title <target-ip-or-domain>
# SSL证书检测
nmap -p 443 --script=ssl-cert <target-ip-or-domain>
# 漏洞扫描
nmap -p 80,443 --script=vuln <target-ip-or-domain>
# HTTP robots.txt检测
nmap -p 80,443 --script=http-robots.txt <target-ip-or-domain>
# SSH主机密钥检测
nmap -p 22 --script=ssh-hostkey <target-ip-or-domain>
```

## 十一、结果处理

```bash
# 将XML结果转换为IP:PORT格式
curl -s 'https://gist.githubusercontent.com/ott3rly/7bd162b1f2de4dcf3d65de07a530a326/raw/83c68d246b857dcf04d88c2db9d54b4b8a4c885a/nmap-xml-to-httpx.sh' | bash -s - nmap.xml
# 将结果传递给httpx，进一步筛选出可访问的服务
curl -s 'https://gist.githubusercontent.com/ott3rly/7bd162b1f2de4dcf3d65de07a530a326/raw/83c68d246b857dcf04d88c2db9d54b4b8a4c885a/nmap-xml-to-httpx.sh' | bash -s - nmap.xml | httpx -mc 200
```

## 十二、扫描IP范围

```bash
# 对于ips.txt中的目标，使用以下命令进行扫描
nmap -iL ips.txt -Pn --min-rate 5000 --max-retries 1 --max-scan-delay 20ms -T4 --top-ports 1000 --exclude-ports 22,80,443,53,5060,8080 --open -oX nmap2.xml
```

