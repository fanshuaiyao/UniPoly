#!/bin/bash
# Uni-Poly 模型训练启动脚本
# 
# 功能说明:
# - 设置Python环境变量，确保可以正确导入项目模块
# - 指定使用的GPU设备
# - 在后台运行训练脚本，并将输出重定向到日志文件
#
# 使用方法:
#   bash scripts/run_train.sh
#   或
#   chmod +x scripts/run_train.sh
#   ./scripts/run_train.sh

# 设置bash解释器
#!/bin/bash

# 设置PYTHONPATH环境变量：将当前目录添加到Python模块搜索路径
# 这样Python可以正确导入src目录下的模块（如src.dataset, src.modules等）
export PYTHONPATH=$(pwd)

# 指定使用的GPU设备：设置CUDA_VISIBLE_DEVICES=1表示使用第2块GPU（索引从0开始）
# 如果只有一块GPU，可以改为 CUDA_VISIBLE_DEVICES=0
# 如果不使用GPU，可以注释掉这一行或设置为空字符串
export CUDA_VISIBLE_DEVICES=1

# 定义训练命令并运行
# nohup: 让命令在后台运行，即使终端关闭也不会中断
# python scripts/train.py: 运行训练脚本
# --modalities: 指定使用的模态（5种模态全部使用）
# --tasks: 指定要训练的任务（5个性质预测任务）
# --pretrained_model_path: 指定预训练模型路径（可选，如果不需要可以删除这一行）
# > ./logs/train.log: 将标准输出重定向到日志文件
# 2>&1: 将标准错误也重定向到同一个日志文件（2是标准错误，1是标准输出）
# &: 在后台运行
nohup python scripts/train.py \
    --modalities smiles text graph fp \
    --tasks lipo \
    > ./logs/train.log 2>&1 &

# 脚本执行后会立即返回，训练在后台进行
# 可以通过以下命令查看训练日志：
#   tail -f ./logs/train.log
# 或查看实时输出：
#   watch -n 1 tail -n 20 ./logs/train.log
