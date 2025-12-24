# 该目录用于存放Dockerfile文件，用于构建docker镜像

## 将Dockerfile 文件移动到/app下

## 构建镜像命令

### 宿主机有网络代理 构建镜像
```bash
docker build --no-cache -f Dockerfile -t jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251224 \
--build-arg http_proxy \
--build-arg https_proxy \
--build-arg HTTP_PROXY \
--build-arg HTTPS_PROXY \
--build-arg no_proxy=localhost,127.0.0.1 \
--build-arg NO_PROXY=localhost,127.0.0.1   .
```

### 宿主机无网络代理 构建镜像
```bash


```

## 运行容器命令

```bash
docker run -d --name asr_offline_server_251224 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/xjtu/model_zoo/model_asr/:/app/model \
  -v /root/config/asr_config_offline.json:/config.json \
  -p 8081:9000 \
  --shm-size=8g \
  jy-algorithm-app-asr-ascend_cann8.2.rc1-ub2204py310:v1.1.9_251224
```