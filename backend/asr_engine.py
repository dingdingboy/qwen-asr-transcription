"""
Qwen ASR inference engine using OpenVINO.
Optimized for Qwen3-ASR models with CPU/GPU/NPU support.
"""

import sys
import logging
from pathlib import Path
from typing import Generator, Optional, List

import numpy as np

# Add path for Qwen3-ASR helper module
HELPER_PATH = Path("/home/aiguru/qwen-asr-transcription/get_convert_qwenasr/Qwen3-ASR")
if str(HELPER_PATH) not in sys.path:
    sys.path.insert(0, str(HELPER_PATH))

# Import OpenVINO ASR model
try:
    from qwen_3_asr_helper import OVQwen3ASRModel, ASRTranscription
    OPENVINO_AVAILABLE = True
except ImportError as e:
    OVQwen3ASRModel = None
    ASRTranscription = None
    OPENVINO_AVAILABLE = False
    logging.warning(f"OpenVINO Qwen3-ASR helper not available: {e}")

# Import inference utilities
try:
    from qwen_asr.inference.utils import SAMPLE_RATE
    INFERENCE_UTILS_AVAILABLE = True
except ImportError:
    SAMPLE_RATE = 16000
    INFERENCE_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class QwenASREngine:
    """
    Qwen ASR inference engine using OpenVINO.
    Supports Qwen3-ASR models with optimized CPU/GPU/NPU inference.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
        n_batch: int = 1,
        verbose: bool = False,
        device: Optional[str] = None,
        quantization: str = "4bit",  # Kept for API compatibility, not used with OpenVINO
        use_gguf: bool = False,       # Kept for API compatibility, not used with OpenVINO
        ov_device: str = "CPU",
        ov_max_batch_size: int = 32,
        ov_max_new_tokens: int = 512,
    ):
        """
        Initialize the OpenVINO-based Qwen ASR engine.

        Args:
            model_path: Path to the converted OpenVINO model directory
            n_ctx: Context size (kept for API compatibility)
            n_threads: Number of threads (kept for API compatibility)
            n_batch: Batch size (kept for API compatibility)
            verbose: Enable verbose logging
            device: Device override (deprecated, use ov_device instead)
            quantization: Quantization mode (kept for API compatibility)
            use_gguf: Use GGUF format (kept for API compatibility)
            ov_device: OpenVINO device to use ("CPU", "GPU", "NPU")
            ov_max_batch_size: Maximum batch size for inference
            ov_max_new_tokens: Maximum new tokens to generate
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError(
                "OpenVINO Qwen3-ASR helper is required. "
                "Please ensure qwen_3_asr_helper.py is available."
            )

        self.model_path = model_path
        self.ov_device = device if device else ov_device
        self.ov_max_batch_size = ov_max_batch_size
        self.ov_max_new_tokens = ov_max_new_tokens
        self.verbose = verbose

        # Legacy attributes for API compatibility
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads else 0
        self.n_batch = n_batch
        self.quantization = quantization
        self.use_gguf = use_gguf

        # Initialize the OpenVINO model
        logger.info(f"Loading OpenVINO model from {model_path}...")
        logger.info(f"Device: {self.ov_device}")
        logger.info(f"Max batch size: {ov_max_batch_size}")

        try:
            self.model = OVQwen3ASRModel.from_pretrained(
                model_dir=model_path,
                device=self.ov_device,
                max_inference_batch_size=ov_max_batch_size,
                max_new_tokens=ov_max_new_tokens,
            )
            logger.info("OpenVINO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        stream: bool = False,
        language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Transcribe a single audio chunk.

        Args:
            audio: Audio data as numpy array (float32, [-1, 1] range, 16kHz)
            stream: If True, yield partial results (not supported with OpenVINO)
            language: Optional language code to force (e.g., "English", "Chinese")

        Yields:
            Transcription text (partial if stream=True, otherwise final)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Validate audio
        if audio is None or len(audio) == 0:
            yield ""
            return

        # Ensure audio is float32 in [-1, 1] range
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        try:
            # Run inference using the OpenVINO model
            # The transcribe method expects audio as (waveform, sample_rate) tuple
            results: List[ASRTranscription] = self.model.transcribe(
                audio=(audio, SAMPLE_RATE),
                context="",  # No context for streaming chunks
                language=language,
                return_time_stamps=False,
            )

            if results and len(results) > 0:
                transcription = results[0]
                text = transcription.text.strip()

                if stream:
                    # For streaming mode, yield the text
                    yield text
                else:
                    # For non-streaming, just yield the final result
                    yield text
            else:
                yield ""

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            yield ""

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Optional language code to force

        Returns:
            Transcription text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            results = self.model.transcribe(
                audio=audio_path,
                context="",
                language=language,
                return_time_stamps=False,
            )

            if results and len(results) > 0:
                return results[0].text.strip()
            return ""

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": self.model_path,
            "device": self.ov_device,
            "ov_max_batch_size": self.ov_max_batch_size,
            "ov_max_new_tokens": self.ov_max_new_tokens,
            "sample_rate": SAMPLE_RATE,
            "backend": "openvino",
        }

        # Add model config info if available
        if self.model and hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "support_languages"):
                info["supported_languages"] = config.support_languages
            if hasattr(config, "model_type"):
                info["model_type"] = config.model_type

        return info

    def __del__(self):
        """Cleanup when the engine is destroyed."""
        # OpenVINO models are automatically cleaned up
        pass