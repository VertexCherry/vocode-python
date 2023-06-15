from enum import Enum
from .model import TypedModel
from enum import EnumMeta
from typing import Optional

class MessageType(str, Enum):
    BASE = "message_base"
    SSML = "message_ssml"
    AUDIO_CLIP = "message_audio_clip"


class BaseMessage(TypedModel, type=MessageType.BASE):
    text: str


class SSMLMessage(BaseMessage, type=MessageType.SSML):
    ssml: str


class AudioClipMessage(BaseMessage, type=MessageType.AUDIO_CLIP):
    audio_clip: str

