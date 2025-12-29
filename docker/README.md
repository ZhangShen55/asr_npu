# 该目录用于存放Dockerfile文件，用于构建docker镜像

## 将Dockerfile 文件移动到/app下

## 构建镜像命令

### 构建基础镜像
```bash
docker build --no-cache -f Dockerfile.8.1rc1cann-py310 -t cann:8.1.rc1-910b-ubuntu22.04-py3.10 \
--build-arg http_proxy \
--build-arg https_proxy \
--build-arg HTTP_PROXY \
--build-arg HTTPS_PROXY \
--build-arg no_proxy=localhost,127.0.0.1 \
--build-arg NO_PROXY=localhost,127.0.0.1   .
```

### 宿主机有网络代理 构建镜像
```bash
docker build --no-cache -f Dockerfile -t jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251226 \
--build-arg http_proxy \
--build-arg https_proxy \
--build-arg HTTP_PROXY \
--build-arg HTTPS_PROXY \
--build-arg no_proxy=localhost,127.0.0.1 \
--build-arg NO_PROXY=localhost,127.0.0.1   .
```

```bash
docker build -f Dockerfile -t jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251226_fix2 \
--build-arg http_proxy \
--build-arg https_proxy \
--build-arg HTTP_PROXY \
--build-arg HTTPS_PROXY \
--build-arg no_proxy=localhost,127.0.0.1 \
--build-arg NO_PROXY=localhost,127.0.0.1   .
```

```bash
docker build -f Dockerfile.no-cython -t jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251227 \
--build-arg http_proxy \
--build-arg https_proxy \
--build-arg HTTP_PROXY \
--build-arg HTTPS_PROXY \
--build-arg no_proxy=localhost,127.0.0.1 \
--build-arg NO_PROXY=localhost,127.0.0.1   .
```
 
# pyarmor
```bash
docker build -f Dockerfile.armor -t jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251227 \
--build-arg http_proxy \
--build-arg https_proxy \
--build-arg HTTP_PROXY \
--build-arg HTTPS_PROXY \
--build-arg no_proxy=localhost,127.0.0.1 \
--build-arg NO_PROXY=localhost,127.0.0.1   .

```
# run

```bash
docker run -d --name asr_offline_server_251227 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/xjtu/model_zoo/model_asr/:/model \
  -v /root/config/asr_config_offline.json:/config.json \
  -e ASCEND_DOCKER_RUNTIME=True \
  -p 8082:9000 \
  --shm-size=32g \
  jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251227
```


```bash
docker run -d --name asr_offline_server_251225   --device /dev/davinci_manager   --device /dev/devmm_svm   --device /dev/hisi_hdc   --device /dev/davinci0   --device /dev/davinci1   -v /usr/local/dcmi:/usr/local/dcmi   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64   -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info   -v /etc/ascend_install.info:/etc/ascend_install.info   -v /home/xjtu/model_zoo/model_asr/:/app/model   -v /root/config/asr_config_offline.json:/config.json  -v /root/config/start.sh:/app/start.sh -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1  --ulimit core=-1 -v /tmp:/tmp  -p 8081:9000   --shm-size=8g   jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251225
```


### 宿主机无网络代理 构建镜像
```bash
docker run -d --name asr_offline_server_251227 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/xjtu/model_zoo/model_asr/:/app/model \
  -v /root/config/asr_config_offline.json:/config.json \
  -e ASCEND_DOCKER_RUNTIME=True \
  -p 8081:9000 \
  --shm-size=128g \
  jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251227
```

## 运行容器命令

```bash
docker run -d --name asr_offline_server_251226_fix2 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/xjtu/model_zoo/model_asr/:/app/model \
  -v /root/config/asr_config_offline.json:/config.json \
  -v /root/config/start.sh:/app/start.sh \
  -v /tmp:/tmp \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONFAULTHANDLER=1  \
  -e ASR_RUN_IN_MAIN_THREAD=1 \
  -e ASCEND_DOCKER_RUNTIME=True \
  -p 8082:9000 \
  --shm-size=128g \
  jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251226_fix2
```


## Dockerfile.cann-py310 文件说明
    - 基于Ubuntu2204，构建Ascend-910b-cann8.2.rc1-ub2204py310环境镜像