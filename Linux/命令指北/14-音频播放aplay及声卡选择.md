###### datetime:2025/07/10 10:10:00

###### author:nzb

# 音频播放aplay

## 命令

```shell
# 列出所有可用声卡
aplay -L 
# 列出所有可用声卡名称
aplay -L | grep :
# 指定声卡播放：plughw 是外接 USB 声卡
speaker-test -D plughw:0,0 -c 2 -t wav
# 或
speaker-test -D plughw:CARD=Device,DEV=0 -c 2 -t wav
# 或
speaker-test -D plughw:Device,0 -c 2 -t wav
# 或
speaker-test -D plughw -c 2 -t wav

# 测试默认设备：
speaker-test -D sysdefault -c 2 -t wav
# 或(可能不起作用)
speaker-test -D default -c 2 -t wav

# 测试 DMIX 设备：
speaker-test -D demixer -c 2 -t wav
```

# 修改配置

找到使用的声卡名称，修改 `/etc/asound.conf` 文件，添加以下内容：

```conf
# 默认 PCM 设备
pcm.!default {
    type plug
    slave {
        pcm "plughw:CARD=Device,DEV=0"  # 使用 plughw 语法
        channels 2
        rate 48000
    }
    hint.description "Device Soundcard"
}

# 默认控制设备
ctl.!default {
    type hw
    card Device  # 使用新声卡名称
}

# DMIX 混音设备
pcm.demixer {
    type plug
    slave {
        pcm {
            type dmix
            ipc_key 1024
            slave {
                pcm "hw:CARD=Device,DEV=0"  # 使用 hw 而非 plughw
                channels 2
                rate 48000
                period_time 0
                period_size 1024
                buffer_size 4096
            }
            bindings {
                0 0
                1 1
            }
        }
    }
}
```

保存后运行 `sudo alsa force-reload` 使配置生效，建议重启系统确保所有服务使用新配置。