from pyannote.core import Segment

def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    words_list = [] # 单句的words列表
    for item in transcribe_res:
        start = item.start
        end = item.end
        text = item.text.strip()
        # if transcribe_res.words is not None:
        words_list = [
                        {
                            "bg":f"{word.start:.2f}",
                            "ed": f"{word.end:.2f}",
                            "word_text" : f"{word.word.strip()}"
                            } for word in (item.words if item and hasattr(item, 'words') and item.words is not None else [])
                            # } for word in item.words   
                            if word is not None]
        timestamp_texts.append((Segment(start, end), text,words_list))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text, words_list in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text, words_list))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-2] for item in text_cache])
    spk = text_cache[0][1]
    start = round(text_cache[0][0].start, 2)
    end = round(text_cache[-1][0].end, 2)
    words_list = [item[-1] for item in text_cache]
    return Segment(start, end), spk, sentence, words_list


PUNC_SENT_END = [',', '.', '?', '!', "，", "。", "？", "！"]


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text, words_list in spk_text:
        # 确保 words_list 有效
        words_list = words_list or []
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            # 当检测到当前说话者与上一个不同时，将缓存中的片段合并并添加到结果列表。
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text,words_list)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            # 如果当前文本片段以句子结束标点（如句号、问号等）结尾，将其添加到缓存后立即合并。
            text_cache.append((seg, spk, text,words_list))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            # 如果既没有说话者变化，也没有遇到句子结束标点，则将当前片段添加到缓存中。
            text_cache.append((seg, spk, text,words_list))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(segments, diarization_result):
    timestamp_texts = get_text_with_timestamp(segments)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed