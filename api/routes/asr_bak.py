import os
import re
import time
import uuid
import shutil
import tempfile
import logging
import asyncio
import torchaudio
import numpy as np
from copy import deepcopy
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException
from fastapi import UploadFile, Form

from app.entity.data import AsrRequestParams, get_asr_params
from app.core.config import settings
from app.core.models import (
    get_asr_model, get_emotion_model, get_online_model, get_whisper_model, get_speaker_model
)
from app.core.concurrency import acquire_gpu_slot, generate_with_gpu_lock, transcribe_with_gpu_lock
from app.utils.audio_utils import preprocess_audio, write_audio_bytes_to_temp_file, crop_audio
from app.utils.pynanote_speaker import diarize_text
from app.utils.feature_utils import extract_features, identify_teacher, convert_role_ids, calculate_speech_rate
from app.utils.asr_stats import update_stat, update_fail_task, add_queued_task, remove_queued_task
from app.utils.character_utils import safe_concat
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

router = APIRouter()


def _generate_task_id() -> str:
    return str(uuid.uuid4())


@router.post("/v1.1.8/seacraft_asr")
async def api_asr_mul(request: AsrRequestParams = Depends(get_asr_params)):
    """
    语音转写 + 小语种(whisper) + 身份判定 + (可选)情绪识别 + (可选)音轨分离
    保持原接口契约不变
    """
    if request.audioFile is None:
        logger.error("音频为空")
        return {"msg": "音频文件不能为空", "code": 4001}

    logger.info(f"request: \n{request}")
    # funasr 识别参数
    param_dict = {
        "batch_size_s": 300,
        "language": "auto",
        "spk_model": "open"
    }
    local_param_dict = deepcopy(param_dict)

    # 处理 hotWords
    if not request.hotWords:
        hotword_str = ""
    else:
        if len(request.hotWords) == 1 and isinstance(request.hotWords[0], str) and "," in request.hotWords[0]:
            request.hotWords = request.hotWords[0].split(",")
        hotword_str = " ".join(request.hotWords)
    local_param_dict["hotword"] = "" if settings.ban_hotword else hotword_str

    # 读取音频
    try:
        start_time = time.perf_counter()
        content = await request.audioFile.read()
        suffix = request.audioFile.filename.split(".")[-1].lower()
        filename = request.audioFile.filename
        if suffix == "m4a":
            suffix = "mp4"
        audio_bytes = await preprocess_audio(content, suffix, force_resample=False)
        if len(audio_bytes) < 1024:
            logger.error(f"音频文件过小，疑似空白音频：{request.audioFile.filename}, 大小：{len(audio_bytes)}")
            update_fail_task()
            raise HTTPException(status_code=400, detail="音频文件过小，疑似空白音频")
        load_audio_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"加载音频文件耗时：{load_audio_time_ms:.2f} ms")
    except Exception as e:
        logger.error(f"读取音频文件发生错误，文件名:{request.audioFile.filename}, 错误信息：\n{e}")
        raise HTTPException(status_code=400, detail=f"读取音频文件发生错误，文件名:{request.audioFile.filename}")


    tmp_paths: List[str] = []
    common_langs = ["fr"]  # 上交大 fr
    audioFile_bytes = content

    # 保存为临时文件（供 whisper / 说话人分离使用）
    tmp_path = write_audio_bytes_to_temp_file(audioFile_bytes, file_name=filename, suffix=f"{suffix}")
    tmp_paths.append(tmp_path)

    # 读取音频 tensor
    audio_tensor, sample_rate = await asyncio.to_thread(torchaudio.load, tmp_path, backend="ffmpeg")

    task_id = f"{_generate_task_id()}_{filename}"

    # 多语种识别不启用写死
    if request.language is None or request.language != "auto":
        request.language = "auto"

    # ---------- 小语种：优先 whisper ----------
    if request.language != "auto" and request.language in common_langs:
        if not settings.open_mul_lang or get_whisper_model() is None:
            logger.error("open_mul_lang 未开启或 whisper 模型未加载")
            update_fail_task()
            return {"msg": "未开启小语种识别或模型未就绪", "code": 4003}

        model_whisper: WhisperModel = get_whisper_model()
        start_gpu = time.time()
        text = ""
        output_segments = []

        async with acquire_gpu_slot(task_id=task_id):
            segments, info = await transcribe_with_gpu_lock(
                model_whisper, tmp_path,
                language=request.language,
                beam_size=5,
                word_timestamps=request.wordTimestamps
            )
        gpu_time_ms = (time.time() - start_gpu) * 1000

        # 有/无音轨分离
        if request.showSpk:
            if not settings.open_mul_spk or get_speaker_model() is None:
                logger.error(f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型 文件名:{filename} 转写失败")
                update_fail_task()
                raise HTTPException(status_code=400, detail=f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型")
                # return {"msg": f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型", "code": 4003}

            speaker_result = get_speaker_model()(tmp_path)
            finally_result = diarize_text(segments, speaker_result)

            # 统计 spk 次数 -> teacher
            spk_id_counts = {}
            for segment, spk, sentence, word_text in finally_result:
                spk_id = int(spk.split("_")[-1]) if spk is not None else 0
                spk_id_counts[spk_id] = spk_id_counts.get(spk_id, 0) + 1
            teacher_id = max(spk_id_counts, key=spk_id_counts.get)
            student_ids = {spk_id: i for i, spk_id in enumerate([i for i in spk_id_counts if i != teacher_id], 1)}

            for segment, spk, sentence, word_text in finally_result:
                spk_id = int(spk.split("_")[-1]) if spk is not None else 0
                role = "teacher" if spk_id == teacher_id else f"student{student_ids[spk_id]}"
                text += sentence + " "
                speed = calculate_speech_rate(sentence, round(segment.start, 2), round(segment.end, 2))
                output_segments.append({
                    "segment_text": sentence,
                    "bg": f"{segment.start:.2f}",
                    "ed": f"{segment.end:.2f}",
                    "speed": speed,
                    "segment_words": [
                        {
                            "bg": f"{float(word['bg']):.2f}",
                            "ed": f"{float(word['ed']):.2f}",
                            "word_text": f"{word['word_text'].strip()}"
                        }
                        for word in (word_text[0] if word_text else []) if word is not None
                    ],
                    "role": role
                })

            # 情绪识别（可选）
            if request.showEmotion:
                if not settings.open_emotion or get_emotion_model() is None:
                    logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename} 转写失败")
                    update_fail_task()
                    return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

                model_emotion = get_emotion_model()
                for seg in output_segments:
                    seg_len_ms = float(seg["ed"]) * 1000 - float(seg["bg"]) * 1000
                    if seg_len_ms > 10000 or seg_len_ms < 1000:
                        seg["emotion"] = "平淡"
                        continue
                    cropped_segment = crop_audio(audio_tensor, float(seg["bg"]) * 1000, float(seg["ed"]) * 1000, sample_rate)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                        seg_path = segf.name
                        tmp_paths.append(seg_path)
                    torchaudio.save(seg_path, cropped_segment, sample_rate)

                    res_emotion = model_emotion.generate(seg_path, granularity="utterance", extract_embedding=False)
                    max_score = max(res_emotion[0]['scores'])
                    emotion_label = res_emotion[0]['labels'][res_emotion[0]['scores'].index(max_score)]
                    if request.noRealEmo:
                        mapping = {
                            "热忱": ["吃惊/surprised", "生气/angry", "恐惧/fearful", "开心/happy"],
                            "平淡": ["<unk>", "其他/other", "中立/neutral"]
                        }
                        if emotion_label in mapping["热忱"]:
                            emotion_label = "热忱"
                        elif emotion_label in mapping["平淡"]:
                            emotion_label = "平淡"
                        emotion = emotion_label
                    else:
                        if emotion_label in ["<unk>", "中立/neutral"] or emotion_label is None:
                            emotion_label = "平淡"
                        emotion = emotion_label.split("/")[0]
                    seg["emotion"] = emotion or "平淡"

            # 清理
            for p in tmp_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

            update_stat("offline")
            return {
                "language": request.language,
                "segments": output_segments,
                "text": text.strip(),
                "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                "gpu_time_ms": f"{gpu_time_ms:.2f}"
            }
        else:
            # 不分离音轨：聚合纯 whisper 输出
            text = ""
            output_segments = []
            for seg in segments:
                text += seg.text.strip() + " "
                output_segments.append({
                    "segment_text": seg.text.strip(),
                    "bg": f"{seg.start:.2f}",
                    "ed": f"{seg.end:.2f}",
                    "segment_words": [
                        {
                            "bg": f"{word.start:.2f}",
                            "ed": f"{word.end:.2f}",
                            "word_text": f"{word.word.strip()}"
                        } for word in (seg.words if seg.words is not None else []) if word is not None
                    ]
                })

            for p in tmp_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            update_stat("offline")
            return {
                "language": request.language,
                "segments": output_segments,
                "text": text.strip(),
                "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                "gpu_time_ms": f"{gpu_time_ms:.2f}"
            }

    # ---------- Paraformer 路线 ----------
    start_time = time.perf_counter()
    model_asr = get_asr_model()

    if request.showSpk:
        if not settings.open_spk or model_asr is None:
            logger.error(f"open_spk={settings.open_spk} 未开启音轨分离模型 文件名:{filename} 转写失败")
            update_fail_task()
            raise HTTPException(status_code=400, detail=f"open_spk={settings.open_spk} 未开启音轨分离模型")
            # return {"msg": f"open_spk={settings.open_spk} 未开启音轨分离模型", "code": 4003}

        try:
            async with acquire_gpu_slot(task_id=task_id):
                rec_results = await generate_with_gpu_lock(
                    model_asr, input=audio_bytes, is_final=True, **local_param_dict
                )
        except (asyncio.TimeoutError, IndexError, HTTPException):
            update_fail_task()
            raise HTTPException(status_code=400, detail="请求过多或超时，请稍后再试")
            # return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

        gpu_time_ms = (time.perf_counter() - start_time) * 1000

        if len(rec_results) == 0:
            update_fail_task()
            return {"text": "", "sentences": [], "code": 0}
        elif len(rec_results) == 1:
            rec_result = rec_results[0]
            text = rec_result["text"]
            if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
                logger.error(f"音频文件为空或未检测到任何人声，可能是静音，文件名:{filename} 转写失败")
                raise HTTPException(status_code=400, detail="音频文件为空或未检测到任何人声")
                # return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

            segments = []
            for segment in rec_result["sentence_info"]:
                segment_words = []
                emotion = None
                if request.wordTimestamps:
                    words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                    timestamps = segment["timestamp"]
                    for i, word_text in enumerate(words):
                        if i < len(timestamps):
                            segment_words.append({
                                "bg": f"{timestamps[i][0] / 1000:.2f}",
                                "ed": f"{timestamps[i][1] / 1000:.2f}",
                                "word_text": word_text
                            })

                if request.showEmotion:
                    if not settings.open_emotion or get_emotion_model() is None:
                        logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename} 转写失败")
                        update_fail_task()
                        raise HTTPException(status_code=400, detail=f"open_emotion={settings.open_emotion} 未开启情感分析模型")
                        # return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

                    emotion_tmp = None
                    # 关键词（文本）优先
                    emotion_dict = {
                        "激昂": [
                            "必须", "一定要", "太重要了", "很重要", "绝对不能", "不错",
                            "请大家认真听", "这特别关键", "很容易错", "太震撼了",
                            "马上就要", "我们要", "这是一种信念", "我再说一遍", "我郑重告诉你们",
                            "太精彩了", "你们一定要记住", "这太关键了", "绝不能忽视",
                            "非常震撼", "简直太强了", "有没有感觉到", "我强调过很多次",
                            "必须掌握", "我强烈建议", "这是重点", "注意这个地方", "这里注意",
                            "这是考试必考", "一定会考的", "这是个大招", "非常核心",
                            "千万别错过", "一不小心就会出错", "我们现在非常重要的一步",
                            "我就问你厉不厉害"
                        ],
                        "热忱": [
                            "我很喜欢", "特别有意思", "真棒", "非常精彩", "我们一起来看一下", "超级有趣",
                            "真好", "太棒了", "超赞", "好有趣", "太可了", "蛮妙的", "很赞",
                            "真的不错", "好理解", "挺喜欢", "我一看就兴奋"
                        ],
                        "沉稳": [
                            "唉", "算了", "随便吧", "就这样吧", "真无语", "懒得说", "不想讲了", "失望",
                            "有点烦", "没啥意思", "有点累", "你们自己看吧", "随你们", "讲不动了",
                            "都讲过了", "不讲也罢", "怎么讲都一样", "讲了也没用", "不清楚？"
                        ]
                    }
                    for emotion_label, keywords in emotion_dict.items():
                        for kw in keywords:
                            if kw in segment["text"]:
                                emotion_tmp = emotion_label
                                break
                        if emotion_tmp:
                            break

                    seg_len_ms = segment["end"] - segment["start"]
                    if emotion_tmp is None and seg_len_ms > 10000:
                        emotion = "平淡"
                    elif emotion_tmp is None:
                        # 音频片段识别
                        cropped = crop_audio(audio_tensor, segment["start"] + 0.1, segment["end"] + 0.1, sample_rate)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                            seg_path = segf.name
                            tmp_paths.append(seg_path)
                        torchaudio.save(seg_path, cropped, sample_rate)
                        res_emotion = get_emotion_model().generate(seg_path, granularity="utterance", extract_embedding=False)
                        max_score = max(res_emotion[0]['scores'])
                        emotion_label = res_emotion[0]['labels'][res_emotion[0]['scores'].index(max_score)]
                        if request.noRealEmo:
                            mapping = {
                                "热忱": ["吃惊/surprised", "生气/angry", "恐惧/fearful", "开心/happy"],
                                "平淡": ["<unk>", "其他/other", "中立/neutral"]
                            }
                            if emotion_label in mapping["热忱"]:
                                emotion_label = "热忱"
                            elif emotion_label in mapping["平淡"]:
                                emotion_label = "平淡"
                            emotion = emotion_label
                        else:
                            if emotion_label in ["<unk>", "中立/neutral"] or emotion_label is None:
                                emotion_label = "平淡"
                            emotion = emotion_label.split("/")[0]
                    else:
                        emotion = emotion_tmp

                speed = calculate_speech_rate(segment["text"], segment["start"] / 1000, segment["end"] / 1000)
                item = {
                    "segment_text": segment["text"],
                    "bg": f"{segment['start'] / 1000:.2f}",
                    "ed": f"{segment['end'] / 1000:.2f}",
                    "speed": speed,
                    "segment_words": segment_words,
                    "role": segment.get("spk")
                }
                if request.showEmotion:
                    item["emotion"] = emotion if emotion is not None else "平淡"
                segments.append(item)

            # 身份识别（老师/学生）
            spk_features = extract_features(segments)
            teacher_role, scores, student_roles = identify_teacher(spk_features)
            if teacher_role is None:
                teacher_role = max(spk_features, key=lambda x: spk_features[x]["keyword_count"])
            segments = convert_role_ids(segments, teacher_role, student_roles)

            ret = {
                "language": request.language,
                "segments": segments,
                "text": text,
                "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                "gpu_time_ms": f"{gpu_time_ms:.2f}",
            }
            update_stat("offline")

            # 清理
            for p in tmp_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            return ret

    # ---------- 不开音轨分离 ----------
    try:
        local_param_dict["spk_model"] = None
        async with acquire_gpu_slot(task_id=task_id):
            rec_results = await generate_with_gpu_lock(
                model_asr, input=audio_bytes, is_final=True, **local_param_dict
            )
    except (asyncio.TimeoutError, IndexError, HTTPException):
        update_fail_task()
        raise HTTPException(status_code=400, detail="请求过多或超时，请稍后再试")
        # return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

    gpu_time_ms = (time.perf_counter() - start_time) * 1000
    if len(rec_results) == 0:
        update_fail_task()
        return {"text": "", "sentences": [], "code": 0}
    elif len(rec_results) == 1:
        rec_result = rec_results[0]
        text = rec_result["text"]
        if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
            logger.error(f"音频文件为空或未检测到任何人声, 可能是静音文件, 文件名:{filename} 转写失败")
            raise HTTPException(status_code=400, detail="音频文件为空或未检测到任何人声,可能是静音文件")
            # return {"msg": "音频文件为空或未检测到任何人声,可能是静音文件", "code": 4008}

        segments = []
        for segment in rec_result["sentence_info"]:
            segment_words = []
            if request.wordTimestamps:
                words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                timestamps = segment["timestamp"]
                for i, word_text in enumerate(words):
                    if i < len(timestamps):
                        segment_words.append({
                            "bg": f"{timestamps[i][0] / 1000:.2f}",
                            "ed": f"{timestamps[i][1] / 1000:.2f}",
                            "word_text": word_text
                        })
            speed = calculate_speech_rate(segment["text"], segment["start"] / 1000, segment["end"] / 1000)
            segments.append({
                "segment_text": segment["text"],
                "bg": f"{segment['start'] / 1000:.2f}",
                "ed": f"{segment['end'] / 1000:.2f}",
                "speed": speed,
                "segment_words": segment_words
            })
        ret = {
            "language": request.language,
            "segments": segments,
            "text": text,
            "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
            "gpu_time_ms": f"{gpu_time_ms:.2f}"
        }
        update_stat("offline")
        for p in tmp_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        return ret
    else:
        update_fail_task()
        for p in tmp_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        return {"msg": "未知错误", "code": 4005}
