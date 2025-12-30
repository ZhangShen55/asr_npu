from funasr import AutoModel
import torch
import torch_npu  # 导入昇腾插件
# 自动迁移：将cuda API映射为npu API
from torch_npu.contrib import transfer_to_npu
import time

print("PyTorch版本:", torch.__version__)
print("NPU设备数量:", torch_npu.npu.device_count())
print("当前NPU设备:", torch_npu.npu.get_device_name(0))
print("Cuda available:", torch.cuda.is_available())

print("开始")

# 使用支持说话人分离的组合模型
# 混合设备配置
model = AutoModel(
    model="paraformer-zh",  # ASR主模型（计算密集型）
    vad_model="fsmn-vad",  # VAD模型（轻量级）
    punc_model="ct-punc",  # 标点模型（轻量级）
    spk_model="cam++",  # 说话人识别模型（计算密集型）
    device="npu:0",  # ASR主模型在npu
    quantize=True,
    batch_size=16,
    disable_update=True,
)


print("开始+++++++++++++++")
for i in range(1):  # 循环100次
    # 输出每次循环的运行时长
    print("模型加载成功")
    start = time.perf_counter()
    res = model.generate(input="/home/xjtu/zhangs/asr_dev/app/test.aac", batch_size_s=300)
    elapsed = time.perf_counter() - start
    # 定义输出数组
    results = []
    # 解析结果
    for result in res:
        sentences = result["sentence_info"]
        for sentence in sentences:
            speaker_id = sentence["spk"]
            text = sentence["text"]
            start_time = sentence["start"]
            end_time = sentence["end"]
            # json 格式输出
            results.append({
                "speakerId": speaker_id,
                "text": text,
                "startTime": start_time,
                "endTime": end_time
            })
            print(f"说话人{speaker_id}: {text} ({start_time:.2f}s-{end_time:.2f}s)")
    print(f"Iteration {i + 1:03d} elapsed: {elapsed:.3f} s")