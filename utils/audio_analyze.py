import numpy as np
import librosa
import os

def segment_audio(y, sr, window_size_sec):
    """切片音频"""
    win_length = int(sr * window_size_sec)
    segments = [y[i:i + win_length] for i in range(0, len(y), win_length)]
    return segments


def calc_db(signal):
    """RMS能量转分贝"""
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return -100.0
    return 20 * np.log10(rms)


def estimate_noise(y, sr, win_len=1):
    """估算静音段能量作为噪声能量（非常粗糙，实际工程可优化）"""
    win_length = int(sr * win_len)
    num_windows = len(y) // win_length
    noise_energies = []
    for i in range(num_windows):
        seg = y[i*win_length:(i+1)*win_length]
        if np.max(np.abs(seg)) < 0.02:  # 低幅值认为是噪声
            noise_energies.append(np.mean(seg**2))
    if noise_energies:
        return np.mean(noise_energies)
    else:
        return 1e-6
    

def calc_snr(signal, noise_energy):
    """信噪比（dB）"""
    sig_energy = np.mean(signal**2)
    if noise_energy == 0:
        return 100
    return 10 * np.log10(sig_energy / noise_energy)


def load_audio_auto(file_path, sr_target=16000, dtype='int16'):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".wav", ".mp3", ".aac"]:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)
        return y, sr
    elif ext == ".pcm":
        y = np.fromfile(file_path, dtype=dtype)
        return y.astype(np.float32) / np.iinfo(dtype).max, sr_target
    else:
        raise ValueError("只支持 wav/mp3/aac/pcm 格式！")

def db_to_volume(db, db_min=-60, db_max=0):
    """分贝转音量（0-100）
    0dB（数字音频最大幅度）：几乎“爆音”/失真，没人会把音频设置这么大。
    -6dB：很响的音量，比如语音主干。
    -20dB：较小的音量，比如远处说话或安静环境。
    -40dB：基本接近静音。
    """
    # 截断到区间
    db = max(min(db, db_max), db_min)
    # 归一化映射到0-100
    return int(round((db - db_min) / (db_max - db_min) * 100))


def analyze_audio_auto(file_path, window_size_sec=1, sr_target=16000):
    y, sr = load_audio_auto(file_path, sr_target)
    segments = segment_audio(y, sr, window_size_sec)
    noise_energy = estimate_noise(y, sr)
    result = []

    # 对每个区间进行处理
    for i, seg in enumerate(segments):
        # 10等分该段音频
        num_subsegments = 10
        subsegment_length = len(seg) // num_subsegments  # 每个子段的长度

        # 用来存放每个子段的db和volume
        subsegment_dbs = []
        subsegment_volumes = []

        # 计算10等分的 db 和 volume
        for j in range(num_subsegments):
            start_idx = j * subsegment_length
            end_idx = (j + 1) * subsegment_length if j != num_subsegments - 1 else len(seg)
            subseg = seg[start_idx:end_idx]

            # 计算每个子段的 db 和 volume
            db = calc_db(subseg)
            volume = db_to_volume(db)

            subsegment_dbs.append(db)
            subsegment_volumes.append(volume)

        # 计算该区间的最大最小和平均值
        max_db = max(subsegment_dbs)
        min_db = min(subsegment_dbs)
        avg_db = np.mean(subsegment_dbs)

        max_volume = max(subsegment_volumes)
        min_volume = min(subsegment_volumes)
        avg_volume = np.mean(subsegment_volumes)

        # 信噪比
        snr = calc_snr(seg, noise_energy)

        result.append({
            "bg": round(float(i * window_size_sec), 1),
            "ed": round(float(min((i + 1) * window_size_sec, len(y) / sr)), 1),
            "max_db": round(float(max_db), 2),  # 最大db
            "db": round(float(avg_db), 2),  # 平均db
            "min_db": round(float(min_db), 2),  # 最小db
            "snr": round(float(snr), 2),
            "max_volume": max_volume,  # 最大volume
            "volume": round(avg_volume),  # 平均volume
            "min_volume": min_volume  # 最小volume
        })

    return result


