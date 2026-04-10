"""
Audio preprocessing utilities for ASR.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioProcessor:
    """Optimized audio preprocessing for ASR."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self._resample_cache = {}

    def preprocess(
        self,
        audio: np.ndarray,
        source_sr: int = 16000,
        normalize: bool = True,
        trim_silence: bool = False
    ) -> np.ndarray:
        """
        Preprocess audio for ASR inference.

        Args:
            audio: Input audio array
            source_sr: Source sample rate
            normalize: Whether to normalize amplitude
            trim_silence: Whether to trim leading/trailing silence

        Returns:
            Preprocessed audio array
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed
        if source_sr != self.target_sr:
            audio = self._resample(audio, source_sr, self.target_sr)

        # Normalize
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        # Trim silence
        if trim_silence and HAS_LIBROSA:
            audio, _ = librosa.effects.trim(audio, top_db=20)

        return audio

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if not HAS_LIBROSA:
            # Simple linear interpolation fallback
            return self._simple_resample(audio, orig_sr, target_sr)

        return librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr
        )

    def _simple_resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    def apply_vad(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, list]:
        """
        Apply simple energy-based VAD.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            frame_duration_ms: Frame duration in milliseconds
            threshold: Energy threshold (0-1)

        Returns:
            Tuple of (speech_segments, timestamps)
        """
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        frames = [
            audio[i:i + frame_size]
            for i in range(0, len(audio), frame_size)
        ]

        # Calculate energy for each frame
        energies = [
            np.sqrt(np.mean(frame ** 2))
            for frame in frames
        ]

        # Normalize energies
        max_energy = max(energies) if energies else 1
        normalized = [e / max_energy for e in energies]

        # Find speech frames
        is_speech = [e > threshold for e in normalized]

        # Group consecutive speech frames
        segments = []
        timestamps = []
        current_start = None

        for i, speech in enumerate(is_speech):
            if speech and current_start is None:
                current_start = i
            elif not speech and current_start is not None:
                # End of speech segment
                start_sample = current_start * frame_size
                end_sample = i * frame_size
                segments.append(audio[start_sample:end_sample])
                timestamps.append({
                    'start': current_start * frame_duration_ms / 1000,
                    'end': i * frame_duration_ms / 1000
                })
                current_start = None

        # Handle trailing speech
        if current_start is not None:
            start_sample = current_start * frame_size
            segments.append(audio[start_sample:])
            timestamps.append({
                'start': current_start * frame_duration_ms / 1000,
                'end': len(frames) * frame_duration_ms / 1000
            })

        if segments:
            return np.concatenate(segments), timestamps
        return np.array([]), []

    def remove_noise(
        self,
        audio: np.ndarray,
        noise_reduction_factor: float = 0.5
    ) -> np.ndarray:
        """
        Simple noise reduction using spectral subtraction.
        Note: This is a basic implementation. For production,
        consider using rnnoise or similar.
        """
        if not HAS_LIBROSA:
            return audio

        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise from first 100ms
        noise_frames = int(0.1 * 16000 / 512)  # ~100ms
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Spectral subtraction
        cleaned_magnitude = np.maximum(
            magnitude - noise_reduction_factor * noise_profile,
            0
        )

        # Reconstruct
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, length=len(audio))

        return cleaned_audio


def pcm_to_float(
    pcm_data: bytes,
    dtype: np.dtype = np.int16
) -> np.ndarray:
    """
    Convert PCM bytes to normalized float32 array.

    Args:
        pcm_data: Raw PCM bytes
        dtype: PCM data type (int16, int32, etc.)

    Returns:
        Normalized float32 array [-1, 1]
    """
    audio = np.frombuffer(pcm_data, dtype=dtype)

    # Normalize based on dtype
    if dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    elif dtype == np.float32:
        return audio
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def float_to_pcm(
    audio: np.ndarray,
    dtype: np.dtype = np.int16
) -> bytes:
    """
    Convert float32 array to PCM bytes.

    Args:
        audio: Float32 array [-1, 1]
        dtype: Target PCM data type

    Returns:
        PCM bytes
    """
    # Clip to prevent overflow
    audio = np.clip(audio, -1, 1)

    if dtype == np.int16:
        pcm = (audio * 32767).astype(np.int16)
    elif dtype == np.int32:
        pcm = (audio * 2147483647).astype(np.int32)
    elif dtype == np.float32:
        pcm = audio.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return pcm.tobytes()
