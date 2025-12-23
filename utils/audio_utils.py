import os
import asyncio
import math
import time
import logging
import torch
from io import BytesIO
import soundfile as sf
from pydub import AudioSegment
# from faster_whisper import WhisperModel
from whisper import Whisper

import tempfile
import subprocess


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("audio")

def check_audio_format(audio_bytes: bytes) -> dict:
    try:
        with sf.SoundFile(BytesIO(audio_bytes)) as f:
            return {
                "samplerate": f.samplerate,
                "channels": f.channels,
                "subtype": f.subtype
            }
    except Exception:
        return {}


def standardize_audio(audio_bytes: bytes, suffix: str, force_resample: bool = False) -> bytes:
    info = check_audio_format(audio_bytes)
    need_resample = force_resample or not (
        info.get("samplerate") == 16000 and
        info.get("channels") == 1 and
        info.get("subtype") == "PCM_16"
    )

    if not need_resample:
        logger.info("[音频检查] 已符合要求，跳过重采样")
        return audio_bytes

    logger.info(f"[音频检查] 转换前参数：{info}，开始重采样")
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format=suffix)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # 导出成 WAV 容器，不能是 raw_data
    buf = BytesIO()
    audio.export(buf, format="wav")          # 默认 PCM_16
    return buf.getvalue()

async def preprocess_audio(audio_bytes: bytes, suffix: str, force_resample: bool = False) -> bytes:
    start = time.perf_counter()
    result = await asyncio.to_thread(standardize_audio, audio_bytes, suffix, force_resample)
    duration = (time.perf_counter() - start) * 1000
    logger.info(f"[音频预处理] 耗时：{duration:.2f}ms")
    return result

# 分割音频
def crop_audio(audio_data:torch.Tensor, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[:,start_sample:end_sample]



def extract_audio_clip(input_path: str, start_time: float, duration: float, suffix=".wav") -> str:
    """
    用 ffmpeg 截取一段音频片段，返回新文件路径
    """
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        output_file
    ]
    subprocess.run(command, check=True)
    return output_file


logger = logging.getLogger(__name__)

def write_audio_bytes_to_temp_file(audio_bytes: bytes, file_name: str, suffix=".mp3") -> str:
    """

    file_name：上游传过来的文件名（带后缀）

    将音频字节写入临时文件，若后缀为 .aac 则转换为 16kHz 单声道 WAV
    
    Args:
        audio_bytes: 音频字节数据
        suffix: 文件后缀，默认为 .mp3
    
    Returns:
        临时文件路径
    """
    if not suffix.startswith("."):
        suffix = "." + suffix
    
    # 保存原始音频到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
        tmp.write(audio_bytes)
        original_file = tmp.name

    # 更改文件名（临时文件是随机名不方便处理） 这里文件名是一个文件路径 不单单是文件名
    new_file_name = f"/tmp/{file_name}"
    os.replace(original_file, new_file_name)
    
    original_file = new_file_name

    # 如果是 aac 格式，转换为 16kHz 单声道 WAV
    if suffix.lower() == ".aac":
        converted_file = original_file.replace(".aac", ".wav")
        
        try:
            # 使用 ffmpeg 进行转换
            subprocess.run([
                "ffmpeg", "-y",  # -y 表示覆盖已存在文件
                "-i", original_file,
                "-ar", "16000",  # 设置采样率为 16kHz
                "-ac", "1",      # 设置为单声道
                "-f", "wav",     # 输出格式为 WAV
                converted_file
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Converted AAC to WAV: {original_file} -> {converted_file}")
            
            # 删除原始 AAC 文件
            os.remove(original_file)
            return converted_file
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert AAC to WAV: {e.stderr.decode()}")
            # 转换失败时返回原始文件
            return original_file
    
    return original_file



def preprocess_audio2wav(input_file: str) -> str:
    # 转换为16k、单通道、wav格式
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    temp_wav = tempfile.mktemp(suffix=".wav")
    audio.export(temp_wav, format="wav")
    return temp_wav


def split_audio(wav_file: str, chunk_size: int):
    audio = AudioSegment.from_file(wav_file)
    duration_ms = len(audio)
    duration_sec = math.ceil(duration_ms / 1000)
    chunks = []

    for i in range(0, duration_ms, chunk_size * 1000):
        chunk = audio[i:i + chunk_size * 1000]
        start_sec = i // 1000
        # 这里判断是否最后一段，end直接用duration_sec，否则正常加
        if i + chunk_size * 1000 >= duration_ms:
            end_sec = duration_sec
        else:
            end_sec = (i + len(chunk)) // 1000
        chunk_path = tempfile.mktemp(suffix=".wav")
        chunk.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_sec, end_sec))
    return chunks


# def detect_language(file_path: str, model: WhisperModel):
def detect_language(file_path: str, model: Whisper):
    # 语言检测
    seg,info = model.transcribe(file_path,language=None)
    # logging.info(f"detect info: {info}")
    lang = info.language
    prob = info.language_probability
    # text = seg.text.strip()
    return lang, prob