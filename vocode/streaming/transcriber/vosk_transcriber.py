import asyncio
import concurrent.futures
import os, ctypes, io, pathlib, wave, wget, json, zipfile, logging, websockets
from pydub import AudioSegment

import numpy as np
from vocode.streaming.agent.utils import SENTENCE_ENDINGS
from vocode.streaming.models.transcriber import VoskTranscriberConfig
from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    BaseThreadAsyncTranscriber,
    Transcription,
)
from vocode.utils.whisper_cpp.helpers import transcribe
from vocode.utils.whisper_cpp.whisper_params import WhisperFullParams
from vosk import Model, SpkModel, KaldiRecognizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VOSK_SAMPLING_RATE = 16000

class VoskLocalServer(object):
    def __init__(self, vosk_model: str):
        self.model = None
        self.spk_model = None
        self.args = None
        self.pool = None

        self.vosk_model_path = f'./{vosk_model}'

        # Check if the model exists, if not, download it.
        model_source = f'https://alphacephei.com/vosk/models/{vosk_model}.zip'
        if not pathlib.Path(self.vosk_model_path).exists():
            print(f'Downloading Vosk model from {model_source}')
            wget.download(model_source, out='./')
            with zipfile.ZipFile(f'{vosk_model}.zip', 'r') as zip_ref:
                zip_ref.extractall('./')
        
    def process_chunk(self, rec, message):
        if message == '{"eof" : 1}':
            return rec.FinalResult(), True
        elif rec.AcceptWaveform(message):
            return rec.Result(), False
        else:
            return rec.PartialResult(), False

    async def recognize(self, websocket, path):
        loop = asyncio.get_running_loop()
        rec = None
        self.phrase_list = None
        self.sample_rate = self.args.sample_rate
        self.show_words = self.args.show_words
        self.max_alternatives = self.args.max_alternatives

        logging.info('Connection from %s', websocket.remote_address)

        while True:

            message = await websocket.recv()

            # Load configuration if provided
            if isinstance(message, str) and 'config' in message:
                jobj = json.loads(message)['config']
                logging.info("Config %s", jobj)
                if 'phrase_list' in jobj:
                    self.phrase_list = jobj['phrase_list']
                if 'sample_rate' in jobj:
                    self.sample_rate = float(jobj['sample_rate'])
                if 'model' in jobj:
                    self.model = Model(jobj['model'])
                    self.model_changed = True
                if 'words' in jobj:
                    self.show_words = bool(jobj['words'])
                if 'max_alternatives' in jobj:
                    self.max_alternatives = int(jobj['max_alternatives'])
                continue

            # Create the recognizer, word list is temporary disabled since not every model supports it
            if not rec or self.model_changed:
                self.model_changed = False
                if self.phrase_list:
                    rec = KaldiRecognizer(self.model, self.sample_rate, json.dumps(self.phrase_list, ensure_ascii=False))
                else:
                    rec = KaldiRecognizer(self.model, self.sample_rate)
                rec.SetWords(self.show_words)
                rec.SetMaxAlternatives(self.max_alternatives)
                if self.spk_model:
                    rec.SetSpkModel(self.spk_model)

            response, stop = await loop.run_in_executor(self.pool, self.process_chunk, rec, message)
            await websocket.send(response)
            if stop: break



    async def start(self):
        #logger.addHandler(logging.StreamHandler())
        #logging.basicConfig(level=logging.INFO)

        self.args = type('', (), {})()

        self.args.interface = os.environ.get('VOSK_SERVER_INTERFACE', '0.0.0.0')
        self.args.port = int(os.environ.get('VOSK_SERVER_PORT', 2700))
        self.args.model_path = os.environ.get('VOSK_MODEL_PATH', 'model')
        self.args.spk_model_path = os.environ.get('VOSK_SPK_MODEL_PATH')
        self.args.sample_rate = float(os.environ.get('VOSK_SAMPLE_RATE', 8000))
        self.args.max_alternatives = int(os.environ.get('VOSK_ALTERNATIVES', 0))
        self.args.show_words = bool(os.environ.get('VOSK_SHOW_WORDS', True))

        # Gpu part, uncomment if vosk-api has gpu support
        #
        # from vosk import GpuInit, GpuInstantiate
        # GpuInit()
        # def thread_init():
        #     GpuInstantiate()
        # pool = concurrent.futures.ThreadPoolExecutor(initializer=thread_init)

        self.model = Model(self.args.model_path)
        self.spk_model = SpkModel(self.args.spk_model_path) if self.args.spk_model_path else None

        self.pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))

        async with websockets.serve(self.recognize, self.args.interface, self.args.port):
            await asyncio.Future()
        
class VoskTranscriber(BaseThreadAsyncTranscriber[VoskTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: VoskTranscriberConfig,
    ):
        super().__init__(transcriber_config)
        self._ended = False
        self.buffer_size = round(
            transcriber_config.sampling_rate * transcriber_config.buffer_size_seconds
        )
        self.buffer = np.empty(self.buffer_size, dtype=np.int16)
        self.buffer_index = 0

        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def create_new_buffer(self):
        buffer = io.BytesIO()
        wav = wave.open(buffer, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(self.transcriber_config.sampling_rate)
        return wav, buffer

    def _run_loop(self):
        in_memory_wav, audio_buffer = self.create_new_buffer()
        message_buffer = ""
        while not self._ended:
            chunk = self.input_janus_queue.sync_q.get()
            in_memory_wav.writeframes(chunk)
            if audio_buffer.tell() >= self.buffer_size * 2:
                audio_buffer.seek(0)
                audio_segment = AudioSegment.from_wav(audio_buffer)
                message, confidence = transcribe(
                    self.whisper, self.params, self.ctx, audio_segment
                )
                message_buffer += message
                is_final = any(
                    message_buffer.endswith(ending) for ending in SENTENCE_ENDINGS
                )
                in_memory_wav, audio_buffer = self.create_new_buffer()
                self.output_queue.put_nowait(
                    Transcription(
                        message=message_buffer, confidence=confidence, is_final=is_final
                    )
                )
                if is_final:
                    message_buffer = ""

    def terminate(self):
        pass
