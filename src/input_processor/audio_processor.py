import justsdk

from .._constants import MODEL_DIR
from ._constants import HF_TOKEN
from faster_whisper import WhisperModel
from pathlib import Path
from pyannote.audio import Pipeline


class AudioProcessor:
    WHISPER_NAME = "whisper-base-en"

    WHISPER_CONFIG = {
        "model_size_or_path": "base.en",
        # "device": "cpu",  # NOTE: Seems like using `auto` is better
        "compute_type": "int8",  # NOTE: `int8` is the smallest quantization
        "num_workers": 2,
        "download_root": str(MODEL_DIR / WHISPER_NAME),
        "local_files_only": True,  # NOTE: Might need to `False` when first init
    }

    AUDIO_CONFIG = {
        "language": "en",
        # "beam_size": 5,  # Paths searches during decoding
        # "best_of": 5,
        # "patience": 1,
        # "length_penalty": 1,
        # "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Temperature fallback
        # "compression_ratio_threshold": 2.4,  # Reject if text is too repetitive
        # "log_prob_threshold": -1.0,  # Threshold for confidence levels
        # "no_speech_threshold": 0.6,  # Threshold for non-speech detection
        "word_timestamps": True,  # Generate word-level timestamps
        "vad_filter": True,  # Skip silent parts
        "vad_parameters": {
            # "threshold": 0.5,
            "min_speech_duration_ms": 250,
            # "max_speech_duration_s": float("inf"),
            # "min_silence_duration_ms": 2000,
            # "speech_pad_ms": 400,
        },
    }

    DIARIZATION_NAME = "pyannote/speaker-diarization-3.1"

    def __init__(self) -> None:
        self.whisper_model = self._init_whisper_model()
        self.diarization_pipeline = self._init_diarization_pipeline()

    def process(self, input_file: Path) -> dict:
        justsdk.print_info(f"Processing audio: {str(input_file)}")
        # transcription = self._transcribe(input_file)

    def _init_whisper_model(self) -> WhisperModel:
        try:
            model = WhisperModel(**self.WHISPER_CONFIG)
            justsdk.print_success(f"Init {self.WHISPER_NAME}", newline_before=True)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.WHISPER_NAME}: {e}")

    def _transcribe(self, audio_file: Path) -> dict:
        try:
            generator, info = self.whisper_model.transcribe(
                audio=str(audio_file), **self.AUDIO_CONFIG
            )
            segments = list(generator)
            result = {
                "segments": segments,
                "info": info,
            }
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe {audio_file}: {e}")

    def _init_diarization_pipeline(self) -> Pipeline:
        try:
            if HF_TOKEN is None:
                raise ValueError("Hugging Face token is not set.")
            pipeline = Pipeline.from_pretrained(
                self.DIARIZATION_NAME, use_auth_token=HF_TOKEN
            )
            # NOTE: Can use `torch` to check if GPU is available
            justsdk.print_success(f"Init {self.DIARIZATION_NAME}", newline_before=True)
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.DIARIZATION_NAME}: {e}")
