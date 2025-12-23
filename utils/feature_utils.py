import re

from app.entity.data import *
from typing import List, Dict, Any, Union
from collections import defaultdict


id2label = {
    0: "what",         # 是何
    1: "why",          # 为何
    2: "how",          # 如何
    3: "what_factors", # 由何
    4: "what_if",      # 若何
    5: "none"           # 非提问句
}
# 教师特征词汇（中英文）
TEACHER_KEYWORDS = [
    # 中文关键词
    "上课", "考试", "上节课", "讲", "我们看", "我们讲", "大家", "同学",
    "请假", "签到", "重点", "考点", "了解一下", "我个人认为",
    "这个非常重要", "要记下来", "这里有", "注意", "这个要考", "下课",
    "休息","我们","一起讨论","回答问题","下次课"

    # 英文关键词
    "class", "lesson", "homework", "exam", "test", "quiz", "assignment",
    "let's discuss", "let's look at", "let's review", "everyone", "students",
    "important point", "key concept", "remember this", "take notes", "pay attention",
    "I want you to", "please focus", "for next class", "as I mentioned", "in our last class"
]

# 英文疑问词
ENGLISH_QUESTION_WORDS = [
    "what", "how", "why", "when", "where", "who", "which", "whose", "whom",
    "can you", "could you", "would you", "will you", "do you", "did you", "have you"
]


def extract_features(segments: List[dict]):
    """提取各个SPK ID的特征"""
    spk_features = defaultdict(lambda: {
        'utterance_count': 0,  # 发言次数
        'total_length': 0,  # 总字数
        'avg_length': 0,  # 平均字数
        'keyword_count': 0,  # 教师关键词出现次数
        'question_count': 0,  # 提问次数
        'speech_time': 0,  # 发言时长
        'segments': []  # 发言时间片段列表
    })

    # 统计基本特征
    for segment in segments:
        # role = segment.role
        # content = segment.segment_text
        # segment_time = float(segment.ed) - float(segment.bg)
        role = segment['role']
        content = segment['segment_text']
        segment_time = float(segment['ed']) - float(segment['bg'])
        # 更新发言次数和总字数
        spk_features[role]['utterance_count'] += 1
        spk_features[role]['total_length'] += len(content)
        spk_features[role]['speech_time'] += segment_time

        # 统计关键词
        for keyword in TEACHER_KEYWORDS:
            if keyword in content:
                spk_features[role]['keyword_count'] += 1

        
        # 中文提问判断（包含）
        # if '？' in content or '?' in content or '吗' in content or '呢' in content:
        #     spk_features[role]['question_count'] += 1
        # 中文提问判断（末尾）
        if content.endswith('？') or content.endswith('?') or content.endswith('吗') or content.endswith('呢'):
            spk_features[role]['question_count'] += 1

        # 英文提问判断
        else:
            content_lower = content.lower()
            # 检查是否以英文疑问词开头或包含英文疑问词
            for question_word in ENGLISH_QUESTION_WORDS:
                if content_lower.startswith(question_word) or f" {question_word} " in content_lower:
                    spk_features[role]['question_count'] += 1
                    break

        # 记录发言时间段
        spk_features[role]['segments'].append(f"{segment['bg']}-{segment['ed']}")

    # 计算平均长度
    for role in spk_features:
        if spk_features[role]['utterance_count'] > 0:
            spk_features[role]['avg_length'] = spk_features[role]['total_length'] / spk_features[role][
                'utterance_count']

    return spk_features



def extract_features_segments(segments: List[Segment]):
    """提取各个SPK ID的特征"""
    spk_features = defaultdict(lambda: {
        'utterance_count': 0,  # 发言次数
        'total_length': 0,  # 总字数
        'avg_length': 0,  # 平均字数
        'keyword_count': 0,  # 教师关键词出现次数
        'question_count': 0,  # 提问次数
        'speech_time': 0,  # 发言时长
        'segments': []  # 发言时间片段列表
    })

    # 统计基本特征
    for segment in segments:
        # role = segment.role
        # content = segment.segment_text
        # segment_time = float(segment.ed) - float(segment.bg)
        role = segment.role
        content = segment.segment_text
        segment_time = float(segment.ed) - float(segment.bg)
        # 更新发言次数和总字数
        spk_features[role]['utterance_count'] += 1
        spk_features[role]['total_length'] += len(content)
        spk_features[role]['speech_time'] += segment_time

        # 统计关键词
        for keyword in TEACHER_KEYWORDS:
            if keyword in content:
                spk_features[role]['keyword_count'] += 1

        
        # 中文提问判断（包含）
        # if '？' in content or '?' in content or '吗' in content or '呢' in content:
        #     spk_features[role]['question_count'] += 1
        # 中文提问判断（末尾）
        if content.endswith('？') or content.endswith('?') or content.endswith('吗') or content.endswith('呢'):
            spk_features[role]['question_count'] += 1

        # 英文提问判断
        else:
            content_lower = content.lower()
            # 检查是否以英文疑问词开头或包含英文疑问词
            for question_word in ENGLISH_QUESTION_WORDS:
                if content_lower.startswith(question_word) or f" {question_word} " in content_lower:
                    spk_features[role]['question_count'] += 1
                    break

        # 记录发言时间段
        spk_features[role]['segments'].append(f"{segment.bg}-{segment.ed}")

    # 计算平均长度
    for role in spk_features:
        if spk_features[role]['utterance_count'] > 0:
            spk_features[role]['avg_length'] = spk_features[role]['total_length'] / spk_features[role][
                'utterance_count']

    return spk_features


# 鉴别老师
def identify_teacher(spk_features):
    """根据特征识别哪个SPK ID是老师"""
    # 计算得分
    scores = {}
    roles = list(spk_features.keys())
    for role, features in spk_features.items():
        # 加权计算得分
        score = (
                features['utterance_count'] * 1.5 +
                features['avg_length'] * 0.5 +
                features['keyword_count'] * 2.0 +
                features['question_count'] * 1.5
        )
        scores[role] = score

    # 找出得分最高的SPK ID
    if scores:
        teacher_role = max(scores, key=scores.get)
        roles_stu = [role for role in roles if role != teacher_role]
        return teacher_role, scores, roles_stu

    return None, {}, []


def merge_consecutive_segments(segments):
    """合并连续的发言时间段"""
    if not segments:
        return []

    # 按开始时间排序
    sorted_segments = sorted(segments, key=lambda x: float(x.split('-')[0]))

    merged = []
    current_start, current_end = map(float, sorted_segments[0].split('-'))

    for segment in sorted_segments[1:]:
        start, end = map(float, segment.split('-'))

        # 如果当前段与上一段连续或重叠
        if start <= current_end + 0.5:  # 允许0.5秒的间隔视为连续
            current_end = max(current_end, end)
        else:
            # 添加当前合并段并开始新段
            merged.append(f"{current_start}-{current_end}")
            current_start, current_end = start, end

    # 添加最后一个合并段
    merged.append(f"{current_start}-{current_end}")

    return merged


def calculate_time_distribution(course_time: float, lecture_time: float, speech_time: float):
    """计算时间分布"""
    # 确保不会出现负值
    freetime = max(0, course_time - lecture_time - speech_time)

    # 防止除以零
    if course_time <= 0:
        return {"lecture": 0, "speech": 0, "freetime": 0}

    return {
        "lecture": round(lecture_time / course_time, 2),
        "speech": round(speech_time / course_time, 2),
        "freetime": round(freetime / course_time, 2)
    }


def merge_segments(segments: List[Segment]) -> List[Dict[str, Any]]:
    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]
        text = current.segment_text.strip()
        bg = current.bg
        ed = current.ed
        spk = current.role

        # 如果当前是逗号结尾，检查下一个是否以 。 或 ？ 结尾且同一个人
        while text.endswith("，") and i + 1 < len(segments):
            next_seg = segments[i + 1]
            next_text = next_seg.segment_text.strip()
            if next_seg.role == spk and (next_text.endswith("。") or next_text.endswith("？")):
                text += next_text
                ed = next_seg.ed
                i += 1
                break # 
            else:
                break

        merged.append({"text": text, "bg": bg, "ed": ed, "role": spk})
        i += 1
    return merged


def format_result(data: List[Dict[str, Any]], target_ids: Union[str, List[str]], speak_time: float,min_len: int = 6) -> Dict[str, Any]:
    # result = {key: {"count": 0, "question_info": {"content": [], "times": []}} for key in id2label.values() if key != "none"}

    # 修正字典推导式
    result = {"role": target_ids , "speak_time": speak_time }
    for key in id2label.values():
        if key != "none":
            result[key] = {"count": 0, "question_info": {"content": [], "times": []}}

    if isinstance(target_ids, str):
        target_ids = [target_ids]

    for entry in data:
        if entry["role"] in target_ids and entry["label"] != "none" and len(entry["text"]) >= min_len:
            q = result[entry["label"]]
            q["count"] += 1
            q["question_info"]["content"].append(entry["text"])
            q["question_info"]["times"].append(f"{entry['bg']}-{entry['ed']}")
    return result



def reformat_result(result_dict):
    # 重构字典结构
    new_result = {
        "role": result_dict["role"],
        "speak_time": result_dict["speak_time"]
    }
    for key in ["what", "why", "how", "what_factors", "what_if"]:
        item = result_dict.get(key, {"count": 0, "question_info": {"content": [], "times": []}})
        qinfo = {c: t for c, t in zip(item.get("question_info", {}).get("content", []),
                                      item.get("question_info", {}).get("times", []))}
        new_result[key] = {
            "count": item.get("count", 0),
            "question_info": qinfo
        }
    return new_result





def convert_role_ids(segments: list, teacher_id: int, student_ids: list) -> list:
    """
    将segments中的数字role ID转换为"teacher"或"studentX"格式的字符串
    
    Args:
        segments: 包含对话片段的列表
        teacher_id: 教师的角色ID
        student_ids: 学生的角色ID列表（可以为空）
    
    Returns:
        修改后的segments列表，其中role字段已转换为字符串格式
    """
    # 创建角色ID到角色名称的映射字典
    role_mapping = {teacher_id: "teacher"}
    # print(f"student_ids: {student_ids}")
    
    # 为每个学生ID分配对应的"studentX"格式
    for index, student_id in enumerate(student_ids):
        role_mapping[student_id] = f"student{index + 1}"
    
    # 处理所有片段
    for segment in segments:
        original_role = segment["role"]
        # 使用get方法提供默认值，确保未映射的ID被标记为unknown
        segment["role"] = role_mapping.get(original_role, f"unknown_{original_role}")
    
    return segments


def calculate_speech_rate(text, start_time, end_time):
    """
    计算语速（字/分钟），区分中文字符和英文单词
    
    参数:
    text (str): 说话内容
    start_time (float): 开始时间（秒）
    end_time (float): 结束时间（秒）
    
    返回:
    int: 语速（字/分钟，取整）
    """
    try:
        # 确保时间差大于0
        duration = end_time - start_time
        if duration <= 0:
            return 120
        
        # 只移除中英逗号、句号、感叹号、问号
        punctuation_to_remove = ",.?!，。！？"
        translator = str.maketrans('', '', punctuation_to_remove)
        clean_text = text.translate(translator)
        
        # 计算中文字符数量
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))
        
        # 计算英文单词数量
        non_chinese_text = re.sub(r'[\u4e00-\u9fff]', '', clean_text)
        english_words = len(non_chinese_text.split())
        
        # 总字数
        total_words = chinese_chars + english_words
        
        # 计算语速（字/分钟），结果取整
        if total_words == 0:
            return 0
            
        duration_minutes = duration / 60
        speech_rate = total_words / duration_minutes
        
        return int(speech_rate * 0.7)  # chang
    
    except Exception as e:
        # print(f"计算语速时出错: {e}")
        return 120