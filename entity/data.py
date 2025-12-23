from dataclasses import dataclass
from fastapi import UploadFile, Depends, Form, File
from typing import List, Optional
from typing import Annotated
# import websockets
from pydantic import BaseModel, Field


@dataclass
class AsrRequestParams:
    audioFile: UploadFile
    wordTimestamps: bool
    hotWords: List[str] = None
    # initialPrompt: Optional[str]
    # vadFilter: bool = False
    # output: Optional[str] = "json"
    language: Optional[str] = "auto"
    # task: Optional[str] = "transcribe"
    showSpk: bool = False
    openPanel: bool = False
    showEmotion: bool = False
    noRealEmo: bool = False
    showSpeed: bool = False
    # ws: Optional[websockets.WebSocketClientProtocol] = None  # 如果使用，需要取消注释

    # def hotwords_as_string(self) -> str:
    #     return ','.join(self.hotWords)


async def get_asr_params(
    audioFile: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form()] = "auto",
    wordTimestamps: Annotated[bool, Form()] = False,
    output: Annotated[Optional[str], Form()] = None,
    hotWords: Annotated[Optional[List[str]], Form()] = None,
    showSpk: Annotated[bool, Form()] = False,
    openPanel: Annotated[bool, Form()] = False,
    showEmotion: Annotated[bool, Form()] = False,
    noRealEmo: Annotated[bool, Form()] = False,
    showSpeed: Annotated[bool, Form()] = False,
) -> AsrRequestParams:
    if hotWords is None:
        hotWords = []
    if output is None:
        output = "json"
    # if initialPrompt is None:
    #     initialPrompt = "请转录为中文简体。"
    return AsrRequestParams(
        audioFile=audioFile,
        # task=task,
        language=language,
        # initialPrompt=initialPrompt,
        # vadFilter=vadFilter,
        wordTimestamps=wordTimestamps,
        # output=output,
        hotWords=hotWords,
        showSpk=showSpk,
        openPanel=openPanel,
        showEmotion=showEmotion,
        noRealEmo=noRealEmo,
        showSpeed=showSpeed
        # ws=ws
    )


class TextSegment(BaseModel):
    segment_text: str
    bg: float  # 开始时间
    ed: float  # 结束时间
    role: str  # 发言者ID


class SpeakerTeacherSegment(BaseModel):
    id: int
    lecture_time_s: Optional[float] = None  # 老师讲授时长
    speech_count: Optional[int] = None  # 说话次数
    # segment: List[str] = None  # 说话时间片段


class SpeakerStudentSegment(BaseModel):
    role: str
    speech_time_s: Optional[float] = None  # 学生发言时长
    speech_count: Optional[int] = None  # 说话次数
    # segment: List[str] = None  # 说话时间片段


# 提问五何分类相关类
class Segment(BaseModel):
    segment_text: str
    bg: str
    ed: str
    segment_words: list | None = None
    role: str
    emotion: str | None = None
    

class SegmentRequestBody(BaseModel):
    segments: List[Segment]
    task_id: Optional[str] = None
    course_id: Optional[str] = None
    confidence: float = 0.6  # nlp置信度阈值
    min_len: int = 10  # 最小句子长度


class CourseInfo(BaseModel):
    '''
    课程互动信息
    '''
    course_time_s: float  # 课程时长
    lecture_time_s: float  # 讲授时长
    speech_time_s: float  # 发言时长
    freetime_time_s: float  # 自由时长
    interaction_times: int  # 互动次数
    average_speech_time_s: float  # 平均发言时长
    time_distribution : dict  # 发言时间分布 lecture speech frertime