import sys
import asyncio
import torch

# =========================================================================
# 1. 关键导入区 (顺序严禁变动)
# =========================================================================

# [规则] ModelScope 必须最先导入，防止 Transformers 抢占 "utils" 命名空间
try:
    # import modelscope.pipelines.base as modelbase
    # import modelscope.utils.device as modeldevice
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    # [新增] 直接导入 FunASR 管道类，解决 "KeyError" 注册表找不到的问题
    # from modelscope.pipelines.audio.funasr_pipeline import FunASRPipeline
except ImportError as e:
    print(f"❌ 环境依赖错误: {e}")
    raise e

# [规则] 接着导入 NPU 支持
try:
    import torch_npu
except ImportError:
    is_npu_available = False
else:
    is_npu_available = True

# [规则] 最后导入其他业务库
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from faster_whisper import WhisperModel
from funasr import AutoModel
import pyannote.audio.core.task

from app.core.config import settings
from app.utils.feature_utils import id2label

print("PyTorch版本:", torch.__version__)
print("NPU设备数量:", torch_npu.npu.device_count())
print("当前NPU设备:", torch_npu.npu.get_device_name(0))
print("Cuda available:", torch.cuda.is_available())

print("开始")


# =========================================================================
# 2. 全局变量与补丁
# =========================================================================

# 单例缓存
_model_asr = None
_model_emotion = None
_model_online = None
_model_whisper = None
_model_speaker = None
_punct_pipeline = None
_model_bert = None
_tokenizer = None

# 线程锁
_model_lock = asyncio.Lock()

# # [NPU 适配补丁] 定义空校验函数
# def _bypass_verify_device(device):
#     pass
#
# # [立即应用补丁] 防止 ModelScope 报错 "device should be cpu/cuda/gpu"
# print("🔧 [NPU适配] 已应用 ModelScope 设备校验补丁")
# modeldevice.verify_device = _bypass_verify_device
# modelbase.verify_device = _bypass_verify_device


# =========================================================================
# 3. 核心功能函数
# =========================================================================

def device() -> torch.device:
    if is_npu_available and torch.npu.is_available():
        # 在 Docker 日志中确认 NPU 是否挂载成功
        # print(f"检测到华为 NPU 设备: {torch.npu.get_device_name(0)}")
        return torch.device(settings.device)
    elif torch.cuda.is_available():
        print(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("未检测到 GPU/NPU 设备，使用 CPU 进行推理")
        return torch.device("cpu")


async def load_models_if_needed():
    """
    根据配置开关懒加载模型。
    """
    global _model_asr, _model_emotion, _model_online, _model_whisper, _model_speaker, _punct_pipeline

    # [NPU 安全] 添加 PyTorch 序列化白名单
    # try:
    #     safe_classes = [
    #         torch.torch_version.TorchVersion,
    #         pyannote.audio.core.task.Specifications,
    #         pyannote.audio.core.task.Problem,
    #         pyannote.audio.core.task.Resolution,
    #     ]
    #     torch.serialization.add_safe_globals(safe_classes)
    #     # print("✅ [NPU安全] 已添加模型加载白名单")
    # except:
    #     pass

    async with _model_lock:
        # 1. 加载 ASR 主模型
        if settings.open_spk and _model_asr is None:
            _model_asr = AutoModel(
                model=settings.asr_model_dir,
                device="npu:0",
                # ngpu=settings.ngpu,
                punc_model=settings.punc_model_dir,
                vad_model=settings.vad_model_dir,
                spk_model=settings.spk_model_dir,
                # vad_kwargs={"max_single_segment_time": 30000, "max_end_silence_time": 800},
                sentence_timestamp=True,
                disable_update=True,
                disable_pbar=False
            )

        # 2. 加载情感分析模型
        if settings.open_emotion and settings.open_spk and _model_emotion is None:
            _model_emotion = AutoModel(
                model=settings.emotion_model_dir,
                device="npu:0",
                ngpu=settings.ngpu,
                disable_update=True,
                disable_pbar=True
            )

        # 3. 加载流式模型
        if settings.open_online and _model_online is None:
            _model_online = AutoModel(
                model=settings.asr_online_model_dir,
                device="npu:0",
                ngpu=settings.ngpu,
                disable_update=True,
                disable_pbar=True
            )

        # 4. 加载 Whisper (注意：CPU 运行)
        if settings.open_mul_lang and _model_whisper is None:
            _model_whisper = WhisperModel(
                settings.whisper_model_dir,
                compute_type=settings.compute_type,
                device="cpu",
                # device="cuda" if torch.cuda.is_available() else "cpu",
                device_index=int(settings.device.split(":")[-1]) if ":" in settings.device else 0
            )

        # 5. 加载标点模型 (使用显式类实例化，解决 Registry 报错)
        if settings.open_online and _punct_pipeline is None:
            pass
            # print("🚀 加载标点模型 (Direct Pipeline Mode)...")
            # _punct_pipeline = pipeline(
            #     model=settings.asr_online_punc_model_dir,
            #     disable_update=True,
            #     device=settings.device
            # )


# =========================================================================
# 4. 辅助 Getter 和 业务逻辑
# =========================================================================

def get_asr_model(): return _model_asr
def get_emotion_model(): return _model_emotion
def get_online_model(): return _model_online
def get_whisper_model(): return _model_whisper
def get_speaker_model(): return _model_speaker
def get_punct_pipeline(): return _punct_pipeline


# ---------- 五何分类 (BERT) ----------
def _ensure_bert_loaded():
    global _model_bert, _tokenizer
    if _model_bert is None or _tokenizer is None:
        _model_bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=settings.bert_model_dir
        ).to(device()).eval()
        _tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=settings.bert_model_tokenizer
        )

def predict_fivewh(text: str) -> tuple[str, int, float]:
    """
    教师提问5何（是何、为何、若何、由何、如何、非提问） bert预测（中文）
    """
    _ensure_bert_loaded()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device())
    with torch.no_grad():
        logits = _model_bert(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return id2label[predicted.item()], predicted.item(), confidence.item()