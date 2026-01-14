#!/bin/bash

# --- 用户配置 ---
# 注意：PUSH_TOKEN在 wechat.py 脚本中修改 注释修改

# GPU占用率阈值 (低于此值视为空闲) 20%
GPU_THRESHOLD=20

# 阶段一：初始空闲时间 (秒) 30min空闲上限制
IDLE_TIME_LIMIT=180

# 阶段二：关机确认时间 (秒)
CONFIRM_TIME_LIMIT=100

# 检查间隔 (秒)
CHECK_INTERVAL=30
# --- 配置结束 ---

# 计算阶段一需要连续检查多少次
IDLE_LIMIT_COUNT=$((IDLE_TIME_LIMIT / CHECK_INTERVAL))
idle_counter=0

# 计算阶段二需要连续检查多少次
CONFIRM_LIMIT_COUNT=$((CONFIRM_TIME_LIMIT / CHECK_INTERVAL))
confirm_counter=0

echo "=== GPU智能监控脚本已启动 ==="
echo "监控配置：GPU占用低于 ${GPU_THRESHOLD}% 持续 ${IDLE_TIME_LIMIT} 秒后发送通知"
echo "通知后：将再持续监控 ${CONFIRM_TIME_LIMIT} 秒，若依旧空闲则关机"
echo "========================="

while true; do
    echo "[$(date)] [监控中] 等待 ${CHECK_INTERVAL} 秒..."
    sleep "$CHECK_INTERVAL"

    # --- 阶段 1: 监控初始空闲 ---
    max_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -nr | head -n 1)

    if [ -z "$max_util" ]; then
        echo "[$(date)] 错误：无法获取GPU信息。脚本暂停 60 秒。"
        sleep 60
        continue # 跳过本次循环
    fi

    echo "[$(date)] [监控中] 当前最高GPU占用: ${max_util}%"

    if [ "$max_util" -lt "$GPU_THRESHOLD" ]; then
        idle_counter=$((idle_counter + 1))
        echo "  > GPU空闲，连续检查次数: ${idle_counter} / ${IDLE_LIMIT_COUNT}"
    else
        if [ "$idle_counter" -gt 0 ]; then
            echo "  > GPU占用恢复，重置计数器。"
        fi
        idle_counter=0
    fi

    # 检查是否达到初始空闲次数
    if [ "$idle_counter" -ge "$IDLE_LIMIT_COUNT" ]; then
        echo "[$(date)] GPU持续空闲已达 ${IDLE_TIME_LIMIT} 秒。"
        
        # 1. 调用 wechat.py 发送通知
        echo "  > 正在调用 wechat.py 发送通知..."
        title="[AutoDL] 训练疑似完成"
        # 更新通知内容，使其与配置(30秒)相匹配
        content="服务器GPU占用低于${GPU_THRESHOLD}%已超过${IDLE_TIME_LIMIT}秒。将开始【关机确认】，若${CONFIRM_TIME_LIMIT}秒后依旧空闲，将自动关机。"
        python wechat.py "${title}" "${content}"
        
        echo "  > 通知已发送。进入 [关机确认] 阶段..."

        # --- 阶段 2: 确认关机 ---
        confirm_counter=0
        idle_counter=0 # 重置阶段1计数器，以便在取消时重新开始

        while [ "$confirm_counter" -lt "$CONFIRM_LIMIT_COUNT" ]; do
            
            echo "[$(date)] [确认中] 等待 ${CHECK_INTERVAL} 秒进行下一次确认..."
            sleep "$CHECK_INTERVAL"

            confirm_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -nr | head -n 1)
            
            echo "[$(date)] [确认中] 当前最高GPU占用: ${confirm_util}%"

            if [ "$confirm_util" -lt "$GPU_THRESHOLD" ]; then
                confirm_counter=$((confirm_counter + 1))
                echo "  > 关机确认中... 进度: ${confirm_counter} / ${CONFIRM_LIMIT_COUNT}"
            else
                echo "[$(date)] GPU占用恢复！取消关机流程，返回 [监控中] 阶段。"
                break # 退出内部循环(阶段2)，返回外部循环(阶段1)
            fi
            
            # 检查是否确认完毕
            if [ "$confirm_counter" -ge "$CONFIRM_LIMIT_COUNT" ]; then
                echo "[$(date)] 关机确认完毕。GPU持续空闲，执行关机。"
                
                # 2. 发送最后一条通知
                title="[AutoDL] 正在关机"
                content="服务器GPU在通知后${CONFIRM_TIME_LIMIT}秒内持续保持空闲。正在执行关机命令。"
                python wechat.py "${title}" "${content}"
                
                # 3. 关机
                echo "[$(date)] 执行关机命令 (shutdown -h now)"
                shutdown -h now
                exit 0 # 成功关机，退出脚本
            fi
            
        done
    fi

done