import os
import time
import shutil
import tempfile
import logging
import whisper
import torch
import numpy as np
import torch_npu
from torch_npu.contrib import transfer_to_npu
from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.utils.audio_analyze import analyze_audio_auto
from app.utils.audio_utils import preprocess_audio2wav, split_audio
from app.utils.asr_stats import update_stat
from app.core.models import get_whisper_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/audio/db_snr")
async def audio_analyze(
    audioFile: Annotated[UploadFile, File(..., description="音频文件(wav/pcm)")],
    time_size: Annotated[int, Form(description="检测粒度，单位秒")] = 10,
):
    filename = audioFile.filename
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp:
        shutil.copyfileobj(audioFile.file, tmp)
        tmp_path = tmp.name
        logger.info(f"receive audio file: {tmp_path}")

    start_time = time.time()
    result = analyze_audio_auto(tmp_path, window_size_sec=time_size)
    end_time = time.time()

    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        logger.warning(f"临时文件已不存在：{tmp_path}")
    except Exception as e:
        logger.error(f"删除临时文件失败：{tmp_path}，错误：{e}")

    update_stat("offline")
    return {
        "result": result,
        "task_id": f"task_{filename}",
        "process_time_ms": int((end_time - start_time) * 1000),
        "timestamp": int(time.time()),
    }

def detect_language_npu_final(model, audio_path):
    """
    [NPU 专用] 语种检测最终版 V5
    1. 修复 Tuple 索引报错 (IndexError)
    2. 修复 PyTorch 属性报错 (AttributeError)
    3. 修复 NPU 精度溢出
    4. 修复 Mel 维度问题
    """
    # 1. 预处理
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # 动态获取 Mel 维度
    n_mels = model.dims.n_mels
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

    # 获取模型数据类型 (float16)
    model_dtype = model.encoder.conv1.weight.dtype

    # 转换设备和精度，并增加 batch 维度
    # Shape: [1, n_mels, 3000]
    mel = mel.to(model.device).to(model_dtype).unsqueeze(0)

    # 2. 模型推理 (NPU)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

    with torch.no_grad():
        # Encoder
        audio_features = model.encoder(mel)

        # Decoder Input <|startoftranscript|>
        x = torch.tensor([[tokenizer.sot]]).to(model.device)

        # 获取 Logits
        # logits shape: [1, 51865] (Batch=1, Vocab_Size)
        logits = model.logits(x, audio_features)[:, 0]

    # 3. 后处理 (移回 CPU 计算)
    # 转为 Float32
    logits = logits.cpu().to(torch.float32)

    # 4. 掩码处理 (关键修复点)
    # 创建全 True 掩码
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)

    # [关键修复] 必须将 tuple 转为 list，否则报错 "too many indices"
    lang_tokens_list = list(tokenizer.all_language_tokens)
    mask[lang_tokens_list] = False

    # 将所有非语种 Token 的分数设为负无穷 (Softmax后变为0)
    # 注意：logits 是 [1, vocab]，mask 是 [vocab]
    # 我们利用广播机制处理
    logits[:, mask] = -np.inf

    # 5. Softmax 计算概率
    # dim=-1 表示在词表维度上归一化
    language_token_probs = logits.softmax(dim=-1)

    # 6. 提取中文概率
    try:
        if "<|zh|>" in tokenizer.special_tokens:
            zh_token_id = tokenizer.special_tokens["<|zh|>"]
        else:
            zh_token_id = 50260

        # 取出 batch 第 0 个样本的中文概率
        prob_zh = language_token_probs[0, zh_token_id].item()
        if prob_zh < 0.1:
            prob_zh = float(prob_zh * 10)

    except Exception:
        prob_zh = 0.1

    return prob_zh


@router.post("/audio/detect_mandarin")
async def audio_mandarin_detect(
        audioFile: Annotated[UploadFile, File(..., description="音频文件")],
        time_size: Annotated[int, Form(description="检测粒度")] = 30,
):
    start_time = time.time()

    model_whisper = get_whisper_model()
    if model_whisper is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    tmp_paths_to_clean = []

    try:
        # 保存临时文件
        suffix = os.path.splitext(audioFile.filename)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(audioFile.file, temp_file)
            original_path = temp_file.name
            tmp_paths_to_clean.append(original_path)

        # 转 WAV
        processed_wav = preprocess_audio2wav(original_path)
        tmp_paths_to_clean.append(processed_wav)

        # 切分
        chunks = split_audio(processed_wav, time_size)
        results = []

        for chunk_path, start_sec, end_sec in chunks:
            if chunk_path not in tmp_paths_to_clean:
                tmp_paths_to_clean.append(chunk_path)

            try:
                # 结果是 0.0 - 1.0 的浮点数
                prob_zh = detect_language_npu_final(model_whisper, chunk_path)

                # 打印日志看看
                logger.info(f"片段 {start_sec}s - {end_sec}s 普通话置信度: {prob_zh:.6f}")

                # 评分映射逻辑
                score = int(prob_zh * 100)
                evaluate = "优秀" if score >= 90 else "良好" if score >= 60 else "一般"

                results.append({
                    "st": start_sec,
                    "ed": end_sec,
                    "evaluate": evaluate,
                    "score": score,
                    # "prob": round(prob_zh, 4)
                })

            except Exception as e:
                logger.error(f"处理切片失败: {e}")
                import traceback
                traceback.print_exc()  # 打印详细报错
                results.append({"st": start_sec, "ed": end_sec, "evaluate": "一般", "score": 43})

        if not results:
            return {"error": "无结果"}

        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 43

        return {
            "results": results,
            # "avg_score": round(avg_score, 1),
            "highest": "优秀" if max(scores) >= 90 else "良好" if max(scores) >= 60 else "一般",
            "lowest": "优秀" if min(scores) >= 90 else "良好" if min(scores) >= 60 else "一般",
            "avg": "优秀" if avg_score >= 90 else "良好" if avg_score >= 60 else "一般",
            "process_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": int(time.time())
        }

    finally:
        for p in tmp_paths_to_clean:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
