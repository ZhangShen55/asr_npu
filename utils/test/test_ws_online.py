
import numpy as np
import asyncio
import websockets

# 假设你的 PCM 文件是 32-bit 单通道音频（4 字节每个样本）
SAMPLE_WIDTH = 2  # 32-bit PCM，1 sample = 4 bytes
CHANNELS = 1       # 单通道
SAMPLE_RATE = 16000  # 16kHz 采样率
chunk_size = 960*8  # 每个块的大小（例如每个块 600ms，即 600ms * 16kHz = 9600 个样本）

async def send_audio_chunks(pcm_file, ws_url):
    try:
        # 读取 PCM 文件
        with open(pcm_file, 'rb') as f:
            pcm_data = f.read()

        # 计算样本总数
        total_samples = len(pcm_data) // SAMPLE_WIDTH  # 每个样本占 2 字节
        print(f"Total samples: {total_samples}")

        async with websockets.connect(ws_url) as websocket:
            # 分块处理 PCM 数据
            for i in range(0, total_samples, chunk_size):
                # 获取当前块的字节数据
                chunk_start = i * SAMPLE_WIDTH
                chunk_end = min((i + chunk_size) * SAMPLE_WIDTH, len(pcm_data))
                chunk = pcm_data[chunk_start:chunk_end]

                # 确保块大小一致，如果最后一块较小，进行填充
                if len(chunk) < chunk_size * SAMPLE_WIDTH:
                    padding_size = (chunk_size * SAMPLE_WIDTH) - len(chunk)
                    chunk += b'\x00' * padding_size  # 用零填充

                try:
                    # 发送音频块
                    await websocket.send(chunk)
                    print(f"Sent chunk {i // chunk_size + 1}")

                    # 接收服务器返回的结果
                    response = await websocket.recv()
                    print(f"Received response for chunk {i // chunk_size + 1}: {response}")

                except Exception as e:
                    print(f"Error while sending/receiving data: {e}")

    except Exception as e:
        print(f"Error processing PCM file: {e}")


async def main_thread():
    # 并发压测
    tasks = []
    for i in range(50):
        pcm_file = f"/home/xjtu/zhangs/asr_dev/test2_16k(1).pcm"  # 可以是同一个文件或不同文件
        # pcm_file = f"/root/workspace/asr_dev/dev/test_file/全程静音.wav"  # 可以是同一个文件或不同文件
        ws_url = "ws://10.236.2.52:8081/v1.0.1/seacraft_asr_online"
        tasks.append(send_audio_chunks(pcm_file, ws_url))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # 多并发压测压测
    asyncio.run(main_thread())


    #======================================================================
    # 单个客户端测试
    # pcm_file = "/home/xjtu/zhangs/asr_dev/test2_16k(1).pcm" # PCM 文件路径
    # ws_url = "ws://10.236.2.52:8081/v1.0.1/seacraft_asr_online"  # WebSocket 服务端地址
    # asyncio.run(send_audio_chunks(pcm_file, ws_url))
