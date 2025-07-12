import justsdk

from configs._constants import MODEL_DIR
from configs._constants import HF_TOKEN
from faster_whisper import WhisperModel
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


class AudioProcessor:
    WHISPER_MODEL_NAME = "whisper-base-en"

    WHISPER_PARAMS = {
        "model_size_or_path": "base.en",
        "compute_type": "int8",
        "num_workers": 2,
        "download_root": str(MODEL_DIR / WHISPER_MODEL_NAME),
        "local_files_only": True,  # NOTE: Set to `False` if the local model is not available
    }

    WHISPER_TRANSCRIBE_PARAMS = {
        "language": "en",
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

    DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-3.1"

    DIARIZATION_PARAM = {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.7,
        }
    }

    def __init__(self) -> None:
        self.whisper_model = self._init_whisper_model()
        self.diarization_pipeline = self._init_diarization_pipeline()

    def process(self, input_file: Path) -> dict:
        justsdk.print_info(f"Processing audio: {str(input_file)}")
        transcription = self._transcribe(input_file)
        diarization = self._diarize(input_file)
        return {
            "transcription": transcription,
            "diarization": diarization,
        }

    def _init_whisper_model(self) -> WhisperModel:
        try:
            model = WhisperModel(**self.WHISPER_PARAMS)
            justsdk.print_success(
                f"Init {self.WHISPER_MODEL_NAME}", newline_before=True
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.WHISPER_MODEL_NAME}: {e}")

    def _transcribe(self, audio_file: Path) -> dict:
        justsdk.print_info(f"Transcribing audio: {str(audio_file)}")
        try:
            generator, info = self.whisper_model.transcribe(
                audio=str(audio_file), **self.WHISPER_TRANSCRIBE_PARAMS
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
                self.DIARIZATION_MODEL_NAME, use_auth_token=HF_TOKEN
            )
            pipeline.instantiate(self.DIARIZATION_PARAM)

            # TODO: Use `torch.cuda.is_available()` to check if GPU is available

            justsdk.print_success(f"Init {self.DIARIZATION_MODEL_NAME}")
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.DIARIZATION_MODEL_NAME}: {e}")

    def _diarize(self, audio_file: Path) -> dict:
        justsdk.print_info(f"Diarizing audio: {str(audio_file)}")
        try:
            with ProgressHook() as hook:
                diarization = self.diarization_pipeline(file=str(audio_file), hook=hook)
                return self._diarize_annotation_to_dict(diarization)
        except Exception as e:
            raise RuntimeError(f"Failed to diarize {audio_file}: {e}")

    def _diarize_annotation_to_dict(self, diarization: any) -> dict:
        result: dict = {
            "segments": [],
            "labels": list(diarization.labels()),
        }
        for segment, _, label in diarization.itertracks(yield_label=True):
            result["segments"].append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.duration,
                    "label": label,
                }
            )
        return result
