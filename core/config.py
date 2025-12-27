import json
import os
from dataclasses import dataclass
from typing import Optional


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Settings:
    # 从环境变量读取 config.json 路径
    config_path: str = os.getenv("CONFIG_PATH", "app/config.json")
    _cfg: dict = None  # 实际配置字典

    def __post_init__(self):
        self._cfg = _load_json(self.config_path)

    # 基础配置
    @property
    def log_path(self) -> str:
        return self._cfg.get("log_path", "./asr_service.log")

    @property
    def id_engine(self) -> str:
        import uuid
        return self._cfg.get("id_engine", f"seacraft-asr-{uuid.uuid4()}")

    @property
    def version(self) -> str:
        return self._cfg.get("version", "seacraft-voice-analysis-app-pro")

    # 设备与并发
    @property
    def ngpu(self) -> int:
        return self._cfg.get("ngpu", 1)

    @property
    def device(self) -> str:
        return self._cfg.get("device", "cuda:0")

    @property
    def ncpu(self) -> int:
        return self._cfg.get("ncpu", 4)

    @property
    def concurrency(self) -> int:
        return self._cfg.get("concurrency", 5)

    @property
    def instance_count(self) -> int:
        return self._cfg.get("instance_count", 4)

    # 模型路径
    @property
    def vad_model_dir(self) -> str:
        return self._cfg.get("vad_model_dir", "/app/model/speech_fsmn_vad_zh-cn-16k-common-pytorch")

    @property
    def punc_model_dir(self) -> str:
        return self._cfg.get("punc_model_dir", "/app/model/punc_ct-transformer_cn-en-common-vocab471067-large")

    @property
    def asr_model_dir(self) -> str:
        return self._cfg.get("asr_model_dir", "/app/model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

    @property
    def spk_model_dir(self) -> Optional[str]:
        return self._cfg.get("spk_model_dir", "/app/model/speech_campplus_sv_zh_en_16k-common_advanced")

    @property
    def emotion_model_dir(self) -> str:
        return self._cfg.get("emotion_modek_dir", "/app/model/emotion2vec_plus_seed")

    @property
    def asr_online_model_dir(self) -> str:
        return self._cfg.get("asr_online_model_dir", "/app/model/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online")

    @property
    def asr_online_punc_model_dir(self) -> str:
        return self._cfg.get("asr_online_punc_model_dir", "/app/model/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")

    @property
    def whisper_model_dir(self) -> str:
        return self._cfg.get("whisper_model_dir", "/model/faster-whisper-large-v3")

    @property
    def compute_type(self) -> str:
        return self._cfg.get("compute_type", "float16")

    @property
    def pyannote_model_yml(self) -> str:
        return self._cfg.get("pyannote_model_yml", "/model/speaker-diarization-3.1/config.yaml")

    # BERT 五何
    @property
    def bert_model_tokenizer(self) -> str:
        return self._cfg.get("bert_model_tokenizer", "/model/bert-base-chinese")

    @property
    def bert_model_dir(self) -> str:
        return self._cfg.get("bert_model_dir", "/model/bert_output/checkpoint-88")

    @property
    def bert_device(self) -> str:
        return self._cfg.get("bert_device", "auto")

    # 功能开关
    @property
    def open_spk(self) -> bool:
        return self._cfg.get("open_spk", False)

    @property
    def open_mul_lang(self) -> bool:
        return self._cfg.get("open_mul_lang", False)

    @property
    def open_mul_spk(self) -> bool:
        return self._cfg.get("open_mul_spk", False)

    @property
    def open_online(self) -> bool:
        return self._cfg.get("open_online", False)

    @property
    def open_emotion(self) -> bool:
        return self._cfg.get("open_emotion", False)

    @property
    def ban_hotword(self) -> bool:
        return self._cfg.get("ban_hotword", False)


settings = Settings()
