import justsdk
import torch

from config._constants import HF_READ_ONLY_TOKEN
from faster_whisper import WhisperModel
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dataclasses import dataclass, field


@dataclass
class AudioProcessorConfig:
    whisper_model_name: str = "whisper-base-en"
    whisper_params: dict = field(
        default_factory=lambda: {
            "model_size_or_path": "base.en",
            "compute_type": "int8",
            "num_workers": 2,
            # "download_root":  # NOTE: Set in `__post_init__`
            # "local_files_only": True,  # NOTE: Just cache it in the default way
        }
    )
    whisper_transcribe_params: dict = field(
        default_factory=lambda: {
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
    )

    diarization_model_name: str = "pyannote/speaker-diarization-3.1"
    diarization_param: dict = field(
        default_factory=lambda: {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.7,
            }
        }
    )

    def __post_init__(self) -> None:
        # self.whisper_params["download_root"] = str(MODEL_DIR / self.whisper_model_name)
        pass


class AudioProcessor:
    def __init__(self) -> None:
        self.acp = AudioProcessorConfig()
        self.whisper_model = self._init_whisper_model()
        self.diarization_pipeline = self._init_diarization_pipeline()

    def process(self, input_file: Path) -> dict:
        """
        Process audio file for transcription and diarization.

        Args:
            input_file: Path to the audio file to process.
        """
        justsdk.print_info(f"Processing audio: {str(input_file)}")
        transcription = self._transcribe(input_file)
        diarization = self._diarize(input_file)
        return {
            "transcription": transcription,
            "diarization": diarization,
        }

    def _init_whisper_model(self) -> WhisperModel:
        try:
            model = WhisperModel(**self.acp.whisper_params)
            justsdk.print_success(
                f"Init {self.acp.whisper_model_name}", newline_before=True
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.acp.whisper_model_name}: {e}")

    def _transcribe(self, audio_file: Path) -> dict:
        justsdk.print_info(f"Transcribing audio: {str(audio_file)}")
        try:
            generator, info = self.whisper_model.transcribe(
                audio=str(audio_file), **self.acp.whisper_transcribe_params
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
            if HF_READ_ONLY_TOKEN is None:
                raise ValueError("Hugging Face token is not set.")
            pipeline = Pipeline.from_pretrained(
                self.acp.diarization_model_name,
                use_auth_token=HF_READ_ONLY_TOKEN,
            )
            pipeline.instantiate(self.acp.diarization_param)

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            justsdk.print_success(f"Init {self.acp.diarization_model_name}")
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.acp.diarization_model_name}: {e}")

    def _diarize(self, audio_file: Path) -> dict:
        justsdk.print_info(f"Diarizing audio: {str(audio_file)}")
        try:
            with ProgressHook() as hook:
                diarization = self.diarization_pipeline(file=str(audio_file), hook=hook)
                return self._diarize_annotation_to_dict(diarization)
        except Exception as e:
            raise RuntimeError(f"Failed to diarize {audio_file}: {e}")

    def _diarize_annotation_to_dict(self, diarization: any) -> dict:
        """
        Convert diarization annotation to a dictionary format.

        Args:
            diarization: The diarization result from PyAnnote.
        """
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
