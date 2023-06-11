from pydub import AudioSegment
from pydantic import BaseModel, validator

class AudioSegmentModel(BaseModel):
    data: bytes

    @validator('data', pre=True)
    def validate_audio(cls, value):
        if isinstance(value, AudioSegment):
            return value.raw_data
        return value

    def to_audio_segment(self):
        return AudioSegment(self.data)