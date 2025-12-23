import subprocess


def convert_audio_to_16k(input_file, output_file):
    try:
        # 构建 ffmpeg 命令
        command = [
            'ffmpeg',
            '-i', input_file,
            '-ar', '16000',
            '-ac', '1',
            output_file
        ]
        # 执行命令
        subprocess.run(command, check=True)
        print(f"音频已成功转换为 16000 采样率，保存为 {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"转换过程中出现错误: {e}")
    except FileNotFoundError:
        print("未找到 ffmpeg，请确保 ffmpeg 已安装并添加到系统环境变量中。")


# 示例调用
input_audio = '/root/workspace/asr_dev/dev/chinEng.wav'
output_audio = '/root/workspace/asr_dev/dev/chinEng-16k.wav'
convert_audio_to_16k(input_audio, output_audio)