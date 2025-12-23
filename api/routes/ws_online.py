import random
import string
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.models import get_online_model, get_punct_pipeline
from app.utils.character_utils import normalize_punct_by_context, safe_concat, count_chinese_and_words
from app.utils.asr_stats import update_stat

router = APIRouter()


@router.websocket("/v1.0.1/seacraft_asr_online")
async def websocket_endpoint3(ws: WebSocket):
    await ws.accept()
    cache = {}

    # chunk 保持一致
    chunk_size = [4, 8, 4]
    encoder_chunk_look_back = 4
    decoder_chunk_look_back = 1
    sampling_points = 960

    is_silence_num = 0
    SILENCE_CHUNKS = 2
    chunk_idx = 0

    buffer = ""
    buffer_bg_time = None

    model_online = get_online_model()
    punctuation = get_punct_pipeline()

    try:
        while True:
            chunk = await ws.receive_bytes()
            speech_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if np.any(~np.isfinite(speech_np)):
                speech_np = np.nan_to_num(speech_np)

            bg = round(chunk_idx * 0.48, 2)
            ed = round((chunk_idx + 1) * 0.48, 2)

            res = model_online.generate(
                input=speech_np,
                cache=cache,
                is_final=False,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )

            if res[0]["text"] == "":
                is_silence_num += 1
            else:
                is_silence_num = 0

            if buffer == "":
                buffer_bg_time = ed
            addition = res[0]["text"]
            buffer = safe_concat(buffer, addition)

            if is_silence_num >= SILENCE_CHUNKS:
                if buffer:
                    buffer = punctuation(buffer, cache={})[0]["text"]
                    buffer = normalize_punct_by_context(buffer)
                await ws.send_json({
                    "key": f"rand_key_{''.join(random.choices(string.ascii_letters + string.digits, k=12))}",
                    "text": buffer,
                    "finished": True if buffer else False,
                    "bg": ed,
                    "ed": round(ed + 0.48, 2)
                })
                cache.clear()
                buffer = ""
                buffer_bg_time = None
            else:
                text = buffer
                if count_chinese_and_words(text) > 200:
                    text = punctuation(text, cache={})[0]["text"]
                    text = normalize_punct_by_context(text)
                    await ws.send_json({
                        "key": f"rand_key_{''.join(random.choices(string.ascii_letters + string.digits, k=12))}",
                        "text": text,
                        "finished": True,
                        "bg": buffer_bg_time,
                        "ed": round(ed + 0.48, 2)
                    })
                    cache.clear()
                    is_silence_num = 0
                    buffer = ""
                    buffer_bg_time = None
                else:
                    if count_chinese_and_words(text) > 3:
                        text = punctuation(text, cache={})[0]["text"]
                        text = normalize_punct_by_context(text)
                        await ws.send_json({
                            "key": f"rand_key_{''.join(random.choices(string.ascii_letters + string.digits, k=12))}",
                            "text": text,
                            "finished": False,
                            "bg": buffer_bg_time,
                            "ed": round(ed + 0.48, 2)
                        })
                    else:
                        await ws.send_json({
                            "key": f"rand_key_{''.join(random.choices(string.ascii_letters + string.digits, k=12))}",
                            "text": text,
                            "finished": False,
                            "bg": buffer_bg_time,
                            "ed": round(ed + 0.48, 2)
                        })
            chunk_idx += 1

    except WebSocketDisconnect:
        pass
    finally:
        update_stat("online")
        if ws.client_state == "CONNECTED":
            await ws.close()
