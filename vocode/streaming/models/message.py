from enum import Enum
from .model import TypedModel
from enum import EnumMeta

from vocode.streaming.models.audio_segment import AudioSegmentModel

class MessageType(str, Enum):
    BASE = "message_base"
    SSML = "message_ssml"
    AUDIO = "message_audio"



class BaseMessage(TypedModel, type=MessageType.BASE):
    text: str


class SSMLMessage(BaseMessage, type=MessageType.SSML):
    ssml: str


class AudioMessage(BaseMessage, type=MessageType.AUDIO):
    audio: AudioSegmentModel
