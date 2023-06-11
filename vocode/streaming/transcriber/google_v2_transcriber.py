import asyncio
import logging
import os
import time
import queue
from typing import Optional
import threading
from vocode import getenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.transcriber.base_transcriber import (
    BaseThreadAsyncTranscriber,
    Transcription,
)
from vocode.streaming.models.transcriber import GoogleV2TranscriberConfig
from vocode.streaming.utils import create_loop_in_thread



def transcribe_streaming_v2(
        project_id: str,
        recognizer_id: str,
        audio_file: str,
) -> cloud_speech.StreamingRecognizeResponse:
    """Transcribes audio from audio file stream.

    Args:
        project_id: The GCP project ID.
        recognizer_id: The ID of the recognizer to use.
        audio_file: The path to the audio file to transcribe.

    Returns:
        The response from the transcribe method.
    """
    # Instantiates a client
    client = SpeechClient()

    request = cloud_speech.CreateRecognizerRequest(
        parent=f"projects/{project_id}/locations/global",
        recognizer_id=recognizer_id,
        recognizer=cloud_speech.Recognizer(
            language_codes=["en-US"], model="latest_long"
        ),
    )

    # Creates a Recognizer
    operation = client.create_recognizer(request=request)
    recognizer = operation.result()

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        content = f.read()

    # In practice, stream should be a generator yielding chunks of audio data
    chunk_length = len(content) // 5
    stream = [
        content[start : start + chunk_length]
        for start in range(0, len(content), chunk_length)
    ]
    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config={})
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config
    )
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=recognizer.name, streaming_config=streaming_config
    )

# TODO: make this nonblocking so it can run in the main thread, see speech.async_client.SpeechAsyncClient
class GoogleV2Transcriber(BaseThreadAsyncTranscriber[GoogleV2TranscriberConfig]):
    def __init__(
        self,
        transcriber_config: GoogleV2TranscriberConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(transcriber_config)

        self.project_id = 'seabassbot'
        self.recognizer_id = 'seabassbot'
        self.language_codes = ["en-US"]
        self.model =  "latest_short"

        self._ended = False
        self.speech_client = SpeechClient()
        self.request = cloud_speech.CreateRecognizerRequest(
            parent=f"projects/{self.project_id}/locations/global",
            recognizer_id=self.recognizer_id,
            recognizer=cloud_speech.Recognizer(
                language_codes=self.language_codes, model=self.model
            ),
        )

        # Creates a Recognizer
        #self.speech_client.update_recognizer()
        #operation = self.speech_client.get_recognizer(request=self.request)
        operation = self.speech_client.create_recognizer(request=self.request)
        self.recognizer = operation.result()

        #credentials_path = getenv("GOOGLE_APPLICATION_CREDENTIALS")
        #if not credentials_path:
        #    raise Exception(
        #        "Please set GOOGLE_APPLICATION_CREDENTIALS environment variable"
        #    )
        #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        #self.google_streaming_config = self.create_google_streaming_config()
        #self.client = self.speech.SpeechClient()
        self.is_ready = False
        if self.transcriber_config.endpointing_config:
            raise Exception("Google V2 endpointing config not supported yet")

    def _run_loop(self):
        stream = self.generator()
        

        audio_requests = (
                cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
            )

        recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config={})
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config
        )

        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer.name, streaming_config=streaming_config
        )
        recognizer_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer.name
        )

        def packets(
                requests: list
        ) -> list:
            first_batch = True
            for audio_request in audio_requests:
                request = (first_batch if [config_request, audio_request] else [recognizer_request, audio_request])
                iterator = ["foo"]#self.speech_client.streaming_recognize(request)
                first_batch = False
                for it in iterator:
                    yield it

        responses_iterator = packets(audio_requests)

        responses = []
        for response in responses_iterator:
            print(f"Transcript: {response}")
            #responses.append(response)
            #for result in response.results:
            #    print(f"Transcript: {result.alternatives[0].transcript}")

        self.process_responses_loop(responses)

    def terminate(self):
        self._ended = True
        super().terminate()

    def process_responses_loop(self, responses):
        for response in responses:
            self._on_response(response)

            if self._ended:
                break

    def _on_response(self, response):
        if not response.results:
            return

        result = response.results[0]
        if not result.alternatives:
            return

        top_choice = result.alternatives[0]
        message = top_choice.transcript
        confidence = top_choice.confidence

        self.output_janus_queue.sync_q.put_nowait(
            Transcription(
                message=message, confidence=confidence, is_final=result.is_final
            )
        )

    def generator(self):
        while not self._ended:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self.input_janus_queue.sync_q.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.input_janus_queue.sync_q.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)
