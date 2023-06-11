import pydub
from pydantic import BaseModel, validator


"""
Contrived example of a special type of date that
takes an int and interprets it as a day in the current year
"""

class AudioSegment(pydub.AudioSegment):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: pydub.AudioSegment):
        return True
