"""
Qwen ASR Transcription Backend
FastAPI application with WebSocket support for real-time audio transcription.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState

from config import get_settings
from asr_engine import QwenASREngine

# Global ASR engine (singleton)
asr_engine: Optional[QwenASREngine] = None

# Active sessions
sessions: Dict[str, "TranscriptionSession"] = {}


class TranscriptionSession:
    """Manages a single transcription session."""

    def __init__(
        self,
        session_id: str,
        asr_engine: QwenASREngine,
        websocket: WebSocket
    ):
        self.session_id = session_id
        self.asr_engine = asr_engine
        self.websocket = websocket
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_duration_ms = 0
        self.max_buffer_ms = 5000  # Process every 500ms
        self.current_segment_id = 0
        self.is_active = True
        self.processing_task: Optional[asyncio.Task] = None

    async def process_audio(self, audio_chunk: np.ndarray):
        """Accumulate and process audio chunks."""
        self.audio_buffer.append(audio_chunk)
        # 16kHz = 16 samples per ms
        self.buffer_duration_ms += len(audio_chunk) / 16

        if self.buffer_duration_ms >= self.max_buffer_ms:
            # Trigger processing without blocking
            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(self._run_inference())

    async def _run_inference(self):
        """Run ASR inference on accumulated buffer."""
        if not self.audio_buffer:
            return

        # Concatenate buffer
        audio = np.concatenate(self.audio_buffer)
        self.audio_buffer = []
        self.buffer_duration_ms = 0

        segment_id = f"seg_{self.current_segment_id}"

        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()

            # Stream partial results
            partial_results = []
            async for partial_text in self._stream_transcription(audio, loop):
                partial_results.append(partial_text)
                await self._send_message({
                    "type": "partial",
                    "text": partial_text,
                    "segmentId": segment_id,
                    "timestamp": asyncio.get_event_loop().time()
                })

            # Send final result
            if partial_results:
                await self._send_message({
                    "type": "final",
                    "text": partial_results[-1],
                    "segmentId": segment_id,
                    "confidence": 0.95,
                    "timestamp": asyncio.get_event_loop().time()
                })

            self.current_segment_id += 1

        except Exception as e:
            await self._send_message({
                "type": "error",
                "code": "INFERENCE_ERROR",
                "message": str(e)
            })

    async def _stream_transcription(
        self,
        audio: np.ndarray,
        loop: asyncio.AbstractEventLoop
    ):
        """Generator that yields partial transcriptions."""

        def run_inference():
            results = []
            for partial in self.asr_engine.transcribe_chunk(audio, stream=True):
                results.append(partial)
            return results

        results = await loop.run_in_executor(None, run_inference)

        for partial in results:
            yield partial

    async def _send_message(self, message: dict):
        """Send message to client."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(message)
        except Exception as e:
            print(f"Failed to send message: {e}")

    async def close(self):
        """Clean up session."""
        self.is_active = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Process any remaining audio
        if self.audio_buffer:
            await self._run_inference()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global asr_engine

    settings = get_settings()

    # Startup: Initialize ASR engine
    print(f"Loading ASR model from {settings.model_path}...")
    print(f"OpenVINO Device: {settings.ov_device}")
    print(f"Max batch size: {settings.ov_max_batch_size}")
    asr_engine = QwenASREngine(
        model_path=settings.model_path,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        verbose=settings.log_level == "debug",
        quantization=settings.quantization,
        use_gguf=settings.use_gguf,
        ov_device=settings.ov_device,
        ov_max_batch_size=settings.ov_max_batch_size,
        ov_max_new_tokens=settings.ov_max_new_tokens,
    )
    print("ASR engine loaded successfully")

    yield

    # Shutdown: Cleanup
    print("Shutting down...")
    for session in list(sessions.values()):
        await session.close()
    sessions.clear()


app = FastAPI(
    title="Qwen ASR Transcription API",
    description="Real-time speech transcription using Qwen ASR with llama.cpp",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": asr_engine is not None,
        "active_sessions": len(sessions)
    }


@app.get("/model-info")
async def model_info():
    """Get model information."""
    if not asr_engine:
        return {"error": "Model not loaded"}

    info = asr_engine.get_model_info()
    return {
        "model_path": info.get("model_path", asr_engine.model_path),
        "backend": info.get("backend", "openvino"),
        "device": info.get("device", asr_engine.ov_device),
        "sample_rate": info.get("sample_rate", 16000),
        "ov_max_batch_size": info.get("ov_max_batch_size", asr_engine.ov_max_batch_size),
        "ov_max_new_tokens": info.get("ov_max_new_tokens", asr_engine.ov_max_new_tokens),
        "supported_languages": info.get("supported_languages", ["English", "Chinese"]),
    }


@app.websocket("/ws/transcribe")
async def transcribe_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription."""
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session: Optional[TranscriptionSession] = None

    try:
        if not asr_engine:
            await websocket.send_json({
                "type": "error",
                "code": "MODEL_NOT_LOADED",
                "message": "ASR model is not available"
            })
            await websocket.close()
            return

        session = TranscriptionSession(session_id, asr_engine, websocket)
        sessions[session_id] = session

        # Send ready message
        model_info = asr_engine.get_model_info() if asr_engine else {}
        await websocket.send_json({
            "type": "ready",
            "sessionId": session_id,
            "modelInfo": {
                "name": "Qwen3-ASR-1.7B",
                "backend": model_info.get("backend", "openvino"),
                "device": model_info.get("device", "CPU"),
                "sampleRate": 16000
            }
        })

        print(f"Session {session_id} started")

        # Message loop
        while session.is_active:
            try:
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary audio data (Int16 PCM)
                    audio_bytes = message["bytes"]
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    # Convert to float32 [-1, 1]
                    audio_float = audio_array.astype(np.float32) / 32768.0

                    await session.process_audio(audio_float)

                elif "text" in message:
                    # Control message
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": data.get("timestamp")
                        })

                    elif msg_type == "end":
                        await session.close()
                        break

                    elif msg_type == "correction":
                        # Handle user correction
                        print(f"Correction received: {data}")
                        # TODO: Store correction for potential training

            except WebSocketDisconnect:
                print(f"Client disconnected: {session_id}")
                break

            except Exception as e:
                print(f"Message handling error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "code": "MESSAGE_ERROR",
                    "message": str(e)
                })

    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": str(e)
            })
        except:
            pass

    finally:
        if session:
            await session.close()
            if session_id in sessions:
                del sessions[session_id]
        try:
            await websocket.close()
        except:
            pass
        print(f"Session {session_id} closed")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.debug
    )
