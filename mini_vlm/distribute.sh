#!/bin/bash

# 用法: ./distribute.sh /path/to/source_dir

SRC_DIR="$1"
DEST_DIR="$1"  # 目标目录与源目录同名
# 远程服务器登录信息

USERNAME="your_username"
PASSWORD="your_password"
PORT=65056

# 目标IP列表
IPS=(
  "172.16.20.14"
  "172.16.20.16"
  "172.16.20.17"
  "172.16.20.22"
)

if [ -z "$SRC_DIR" ]; then
  echo "请传入数据源目录作为参数"
  exit 1
fi

# 检查 sshpass 是否安装
if ! command -v sshpass &> /dev/null; then
  echo "请先安装 sshpass 工具：sudo apt install sshpass"
  exit 1
fi

for IP in "${IPS[@]}"; do
  echo "正在向 $IP 分发..."
  # 先远程创建目录
  sshpass -p "$PASSWORD" ssh -p $PORT -o StrictHostKeyChecking=no "$USERNAME@$IP" "mkdir -p \"$DEST_DIR\""
  # 分发数据
  sshpass -p "$PASSWORD" scp -P $PORT -o StrictHostKeyChecking=no -r "$SRC_DIR" "$USERNAME@$IP:$DEST_DIR"
  if [ $? -eq 0 ]; then
    echo "$IP 分发成功"
  else
    echo "$IP 分发失败"
  fi
done