import sys
import asyncio
import importlib.util
import os

# =========================================================================
# 🚑 紧急环境清洗器 (Namespace Scrubber)
# 必须放在任何其他 import 之前执行！
# =========================================================================
print("🧹 [System] Starting Pre-flight Namespace Check...")

# 1. 强制清理 transformers.utils 对 modelscope.utils 的污染
# 如果内存中 modelscope.utils 已经被指向了 transformers.utils，强制删除它
conflicted_modules = ['modelscope.utils', 'modelscope.utils.device']
for mod_name in conflicted_modules:
    if mod_name in sys.modules:
        # 获取当前模块对象
        mod = sys.modules[mod_name]
        # 检查它是否实际上是 transformers 的一部分
        if 'transformers' in str(mod):
            print(f"🚨 Detected collision: {mod_name} points to {mod} -> DELETING")
            del sys.modules[mod_name]

# 2. 尝试手动从文件路径加载 modelscope.utils.device
# 绕过 Python 混乱的 import 缓存机制
try:
    import modelscope

    # 获取 modelscope 包的安装路径
    ms_path = os.path.dirname(modelscope.__file__)
    device_py_path = os.path.join(ms_path, 'utils', 'device.py')

    if os.path.exists(device_py_path):
        spec = importlib.util.spec_from_file_location("modelscope.utils.device", device_py_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["modelscope.utils.device"] = module
            spec.loader.exec_module(module)
            print("✅ Manually loaded modelscope.utils.device from file system.")
    else:
        print(f"⚠️ Warning: Could not find {device_py_path}")

except Exception as e:
    print(f"⚠️ Manual load failed: {e}. Falling back to standard import.")

# =========================================================================
# 🏁 正常导入流程 (带补丁)
# =========================================================================

try:
    # 定义空校验函数
    def _bypass_verify_device(device):
        pass


    # 导入 modelscope 组件
    import modelscope.pipelines.base as modelbase
    import modelscope.utils.device as modeldevice  # 现在这行应该能正常工作了
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    # ⚡️ 立即应用 NPU 适配补丁
    print("🔧 Applying NPU Patch to ModelScope...")
    modeldevice.verify_device = _bypass_verify_device
    modelbase.verify_device = _bypass_verify_device

    # 双重保险：直接修改 sys.modules 里的对象
    if 'modelscope.utils.device' in sys.modules:
        sys.modules['modelscope.utils.device'].verify_device = _bypass_verify_device

except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    # 打印详细调试信息
    if 'modelscope.utils' in sys.modules:
        print(f"DEBUG: modelscope.utils points to: {sys.modules['modelscope.utils']}")
    raise e

# --- NPU 检测 ---
import torch

try:
    import torch_npu
except ImportError:
    is_npu_available = False
else:
    is_npu_available = True

# --- 其他库导入 (Transformers 必须放在这里，不能提早) ---
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from faster_whisper import WhisperModel
from funasr import AutoModel
import pyannote.audio.core.task

from app.core.config import settings
from app.utils.feature_utils import id2label

# --- 业务逻辑 ---
_model_asr = None
_model_emotion = None
_model_online = None
_model_whisper = None
_model_speaker = None
_punct_pipeline = None
_model_bert = None
_tokenizer = None
_model_lock = asyncio.Lock()


def device() -> torch.device:
    if is_npu_available and torch.npu.is_available():
        print(f"检测到华为 NPU 设备: {torch.npu.get_device_name(0)}")
        return torch.device(settings.device)
    elif torch.cuda.is_available():
        print(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("未检测到 GPU 设备，使用 CPU 进行推理")
        return torch.device("cpu")

async def load_models_if_needed():
    """
    根据配置开关懒加载模型。
    """
    global _model_asr, _model_emotion, _model_online, _model_whisper, _model_speaker, _punct_pipeline

    # 再次确认补丁生效（防止运行时被覆盖）
    try:
        modeldevice.verify_device = _bypass_verify_device
        modelbase.verify_device = _bypass_verify_device
    except:
        pass

    # PyTorch 白名单
    try:
        safe_classes = [
            torch.torch_version.TorchVersion,
            pyannote.audio.core.task.Specifications,
            pyannote.audio.core.task.Problem,
            pyannote.audio.core.task.Resolution,
        ]
        torch.serialization.add_safe_globals(safe_classes)
        print("✅ 已成功添加模型安全白名单")
    except:
        pass

    async with _model_lock:
        if settings.open_spk and _model_asr is None:
            _model_asr = AutoModel(
                model=settings.asr_model_dir,
                device=settings.device,
                ngpu=settings.ngpu,
                punc_model=settings.punc_model_dir,
                vad_model=settings.vad_model_dir,
                spk_model=settings.spk_model_dir,
                vad_kwargs={"max_single_segment_time": 30000, "max_end_silence_time": 800},
                sentence_timestamp=True,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_emotion and settings.open_spk and _model_emotion is None:
            _model_emotion = AutoModel(
                model=settings.emotion_model_dir,
                device=settings.device,
                ngpu=settings.ngpu,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_online and _model_online is None:
            _model_online = AutoModel(
                model=settings.asr_online_model_dir,
                device=settings.device,
                ngpu=settings.ngpu,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_mul_lang and _model_whisper is None:
            _model_whisper = WhisperModel(
                settings.whisper_model_dir,
                compute_type=settings.compute_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
                device_index=int(settings.device.split(":")[-1]) if ":" in settings.device else 0
            )


        if settings.open_online and _punct_pipeline is None:
            pass
            # _punct_pipeline = pipeline(
            #     task=Tasks.punctuation,
            #     model=settings.asr_online_punc_model_dir,
            #     disable_update=True,
            #     device=settings.device
            # )



def get_asr_model():
    return _model_asr


def get_emotion_model():
    return _model_emotion


def get_online_model():
    return _model_online


def get_whisper_model():
    return _model_whisper


def get_speaker_model():
    return _model_speaker


def get_punct_pipeline():
    return _punct_pipeline


# ---------- 五何分类 ----------
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
