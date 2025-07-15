# 运行 Docker 容器，挂载数据目录并启用 GPU
docker container run --gpus "device=1" -m 28G --name baseline_xjtu_dgddm --rm \
-v $PWD/FLARE_Test/:/workspace/inputs/ \
-v $PWD/baseline_xjtu_dgddm/:/workspace/outputs/ \
baseline_xjtu_dgddm:latest /bin/bash -c "sh predict.sh"

