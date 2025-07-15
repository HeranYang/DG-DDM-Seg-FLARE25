#!/bin/bash

# 循环运行10次训练命令
for i in {1..20}
do
    echo "第 $i 次训练开始..."
    python sr_train.py -p train -c config/SR3_EffNet_train2.json
    echo "第 $i 次训练完成。"
done
