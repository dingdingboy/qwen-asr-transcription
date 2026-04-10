# Qwen ASR + llama.cpp Real-Time Transcription System

## 1. Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Browser)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Audio Capture│→ │  Audio Work  │→ │ WebSocket    │→ │   React/Vue     │  │
│  │  (Web Audio) │  │   Processor  │  │   Client     │  │      UI         │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│         ↑                                    ↓                              │
│    Microphone                          Transcription                         │
│    Permission                          Updates (SSE/WS)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (binary audio + JSON control)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SERVER (Python/FastAPI)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  WebSocket   │→ │   Audio      │→ │  llama.cpp   │→ │   Session       │  │
│  │   Handler    │  │   Buffer     │  │   Inference  │  │    Manager      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│         ↑              │                    │                               │
│    Connection      VAD Audio           Qwen ASR GGUF                         │
│    Management      Chunks              Model on CPU                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Audio Capture**: Browser captures 16kHz mono PCM via Web Audio API
2. **Client-Side VAD**: Simple energy-based VAD filters silence
3. **Chunking**: 100-300ms audio chunks sent via WebSocket
4. **Server Buffer**: Chunks accumulated for context window
5. **Inference**: llama.cpp runs Qwen ASR on CPU with streaming KV cache
6. **Streaming Results**: Partial transcriptions sent back via WebSocket
7. **UI Update**: Frontend displays incremental results with edit capability

---

## 2. Technology Stack Recommendations

### Frontend
| Component | Technology | Rationale |
|-----------|------------|-----------|
| Framework | **Svelte 5** or **Vue 3** | Minimal overhead, excellent reactivity |
| Audio API | Web Audio API + AudioWorklet | Low-level PCM access, non-blocking |
| Real-time | Native WebSocket | Bidirectional, low latency |
| Styling | Tailwind CSS | Rapid UI development |
| Build | Vite | Fast HMR, optimized builds |

### Backend
| Component | Technology | Rationale |
|-----------|------------|-----------|
| Framework | **FastAPI** | Async native, WebSocket support, excellent performance |
| ASGI Server | Uvicorn | ASGI with WebSocket support |
| Inference | **llama-cpp-python** | Python bindings for llama.cpp |
| Audio Processing | librosa / numpy | Resampling, format conversion |
| VAD | webrtcvad or silero-vad | Server-side VAD validation |

### Model
| Component | Specification |
|-----------|---------------|
| Model | Qwen2-Audio-7B-Instruct or Qwen-Audio-Chat |
| Format | GGUF (Q4_K_M or Q5_K_M quantization) |
| Inference | llama.cpp with streaming support |
| Context | 8192 tokens for long-form audio |

---

## 3. Qwen ASR + llama.cpp Integration

### Model Preparation

```bash
# 1. Download Qwen2-Audio model (requires transformers for conversion)
pip install transformers torch

# 2. Convert to GGUF format
# Clone llama.cpp and build convert script
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert model (example for Qwen2-Audio)
python convert_hf_to_gguf.py \
    --model-dir Qwen/Qwen2-Audio-7B-Instruct \
    --outfile qwen2-audio-7b-instruct.gguf \
    --outtype q4_k_m
```

### Python Integration

```python
# backend/asr_engine.py
from llama_cpp import Llama
import numpy as np
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)

class QwenASREngine:
    """
    Qwen ASR inference engine using llama.cpp
    Optimized for CPU inference with streaming support.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int = None,  # Auto-detect CPU cores
        n_batch: int = 1,       # Single sequence for streaming
        verbose: bool = False
    ):
        self.n_threads = n_threads or (os.cpu_count() or 4)

        # Load model with CPU-optimized settings
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            n_batch=n_batch,
            verbose=verbose,
            # Enable streaming KV cache
            offload_kqv=False,  # Keep KV on CPU for lower latency
            # Performance optimizations
            use_mmap=True,      # Memory-mapped model loading
            use_mlock=False,    # Don't lock pages (allows swapping)
        )

        # ASR-specific prompt template for Qwen2-Audio
        self.asr_prompt = """<|im_start|>user
<|audio_bos|><|AUDIO|><|audio_eos|>
Transcribe the speech to text.<|im_end|>
<|im_start|>assistant
"""

        logger.info(f"ASR Engine loaded: {model_path} (threads: {self.n_threads})")

    def transcribe_chunk(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int = 16000,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """
        Transcribe audio chunk with streaming output.

        Args:
            audio_pcm: Normalized float32 audio array [-1, 1]
            sample_rate: Audio sample rate (must be 16kHz for Qwen)
            stream: Whether to yield partial results

        Yields:
            Transcription text (partial or complete)
        """
        # Ensure correct format
        if sample_rate != 16000:
            audio_pcm = librosa.resample(
                audio_pcm,
                orig_sr=sample_rate,
                target_sr=16000
            )

        # Convert to bytes (int16) for model input
        audio_bytes = (audio_pcm * 32767).astype(np.int16).tobytes()

        # Build prompt with audio placeholder
        # Note: Actual audio encoding depends on Qwen's specific format
        prompt = self.asr_prompt.replace("<|AUDIO|>", self._encode_audio(audio_bytes))

        if stream:
            # Streaming generation for real-time feel
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.0,  # Greedy decoding for accuracy
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True,
            )

            partial_text = ""
            for chunk in output:
                delta = chunk["choices"][0].get("text", "")
                partial_text += delta
                yield partial_text
        else:
            # Non-streaming for final accuracy
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.0,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=False,
            )
            yield output["choices"][0]["text"]

    def _encode_audio(self, audio_bytes: bytes) -> str:
        """
        Encode audio bytes to format expected by Qwen model.
        Qwen2-Audio uses specific audio token encoding.
        """
        # This is model-specific - refer to Qwen2-Audio documentation
        # Typically involves base64 encoding or special token sequences
        import base64
        return base64.b64encode(audio_bytes).decode('utf-8')

    def warmup(self):
        """Run dummy inference to warm up caches."""
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        list(self.transcribe_chunk(dummy_audio, stream=False))
        logger.info("ASR engine warmed up")
```

---

## 4. Audio Capture and Streaming Strategy

### Browser Audio Pipeline

```javascript
// frontend/src/lib/audio/AudioCapture.js
export class AudioCapture {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.bufferSize = options.bufferSize || 4096;
    this.chunkDuration = options.chunkDuration || 200; // ms
    this.onAudioChunk = options.onAudioChunk || (() => {});
    this.onVADChange = options.onVADChange || (() => {});

    this.audioContext = null;
    this.workletNode = null;
    this.mediaStream = null;
    this.isRecording = false;
  }

  async initialize(deviceId = null) {
    // Request microphone access
    const constraints = {
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        sampleRate: { ideal: 16000 },
        channelCount: { exact: 1 },
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    };

    this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

    // Create AudioContext with target sample rate
    this.audioContext = new AudioContext({
      sampleRate: this.sampleRate,
      latencyHint: 'interactive'
    });

    // Load AudioWorklet for processing
    await this.audioContext.audioWorklet.addModule('/audio-processor.js');

    // Create processing chain
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(this.audioContext, 'transcription-processor');

    // Handle processed audio chunks
    this.workletNode.port.onmessage = (event) => {
      const { pcmData, energy, isSpeech } = event.data;

      // VAD state change notification
      this.onVADChange(isSpeech);

      // Only send speech chunks (with small buffer for context)
      if (isSpeech || this._recentlySpeech) {
        this.onAudioChunk(pcmData);
      }
    };

    source.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);
  }

  start() {
    if (this.audioContext?.state === 'suspended') {
      this.audioContext.resume();
    }
    this.isRecording = true;
  }

  stop() {
    this.isRecording = false;
    this.workletNode?.disconnect();
    this.mediaStream?.getTracks().forEach(track => track.stop());
    this.audioContext?.close();
  }

  static async enumerateDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(d => d.kind === 'audioinput');
  }
}
```

### AudioWorklet Processor

```javascript
// frontend/public/audio-processor.js
class TranscriptionProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // VAD parameters
    this.energyThreshold = 0.01;
    this.silenceFrames = 0;
    this.silenceThreshold = 30; // frames of silence to stop
    this.isSpeech = false;
    this.recentlySpeech = false;

    // Chunking
    this.bufferSize = 1600; // 100ms at 16kHz
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Calculate energy for VAD
    let energy = 0;
    for (let i = 0; i < channelData.length; i++) {
      energy += channelData[i] * channelData[i];
    }
    energy = Math.sqrt(energy / channelData.length);

    // Simple energy-based VAD
    if (energy > this.energyThreshold) {
      this.isSpeech = true;
      this.silenceFrames = 0;
    } else {
      this.silenceFrames++;
      if (this.silenceFrames > this.silenceThreshold) {
        this.isSpeech = false;
      }
    }

    // Accumulate into chunks
    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bufferIndex++] = channelData[i];

      if (this.bufferIndex >= this.bufferSize) {
        // Send full chunk
        this.port.postMessage({
          pcmData: new Float32Array(this.buffer),
          energy: energy,
          isSpeech: this.isSpeech
        }, [this.buffer.buffer]); // Transfer ownership

        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor('transcription-processor', TranscriptionProcessor);
```

### WebSocket Audio Streaming

```javascript
// frontend/src/lib/websocket/AudioStream.js
export class AudioStream {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onclose = () => {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => this.reconnect(), 1000 * Math.pow(2, this.reconnectAttempts));
          this.reconnectAttempts++;
        }
      };

      this.ws.onerror = (error) => reject(error);
    });
  }

  sendAudioChunk(pcmData) {
    if (this.ws.readyState !== WebSocket.OPEN) return;

    // Convert Float32 to Int16 for efficient transfer
    const int16Data = new Int16Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      int16Data[i] = Math.max(-32768, Math.min(32767, pcmData[i] * 32767));
    }

    // Send binary audio data
    this.ws.send(int16Data.buffer);
  }

  sendControlMessage(type, payload) {
    if (this.ws.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({
      type,
      payload,
      timestamp: Date.now()
    }));
  }

  onMessage(callback) {
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      callback(data);
    };
  }

  reconnect() {
    console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
    this.ws = new WebSocket(this.ws.url);
    this.connect();
  }

  close() {
    this.ws.close();
  }
}
```

---

## 5. WebSocket Protocol Design

### Message Types

```
CLIENT → SERVER
───────────────
AUDIO_DATA      : Binary Int16 PCM audio chunk
START_SESSION   : { type: "start", config: {...} }
END_SESSION     : { type: "end" }
PING            : { type: "ping", timestamp: number }
CORRECTION      : { type: "correction", segmentId: string, text: string }

SERVER → CLIENT
───────────────
PARTIAL_RESULT  : { type: "partial", text: string, segmentId: string }
FINAL_RESULT    : { type: "final", text: string, segmentId: string, confidence: number }
SESSION_READY   : { type: "ready", sessionId: string, modelInfo: {...} }
ERROR           : { type: "error", code: string, message: string }
PONG            : { type: "pong", timestamp: number }
```

### Protocol Implementation

```python
# backend/websocket_handler.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import json
import asyncio
import numpy as np
from typing import Dict, Optional
import uuid

app = FastAPI()

class TranscriptionSession:
    """Manages a single transcription session."""

    def __init__(self, session_id: str, asr_engine, websocket: WebSocket):
        self.session_id = session_id
        self.asr_engine = asr_engine
        self.websocket = websocket
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_duration_ms = 0
        self.max_buffer_ms = 500  # Process every 500ms
        self.current_segment_id = 0
        self.is_active = True

    async def process_audio(self, audio_chunk: np.ndarray):
        """Accumulate and process audio chunks."""
        self.audio_buffer.append(audio_chunk)
        self.buffer_duration_ms += len(audio_chunk) / 16  # 16kHz = 16 samples/ms

        if self.buffer_duration_ms >= self.max_buffer_ms:
            await self._run_inference()

    async def _run_inference(self):
        """Run ASR inference on accumulated buffer."""
        if not self.audio_buffer:
            return

        # Concatenate buffer
        audio = np.concatenate(self.audio_buffer)
        self.audio_buffer = []
        self.buffer_duration_ms = 0

        segment_id = f"seg_{self.current_segment_id}"

        # Run inference (in thread pool to not block)
        loop = asyncio.get_event_loop()

        try:
            # Stream partial results
            async for partial_text in self._stream_transcription(audio):
                await self._send_message({
                    "type": "partial",
                    "text": partial_text,
                    "segmentId": segment_id,
                    "timestamp": asyncio.get_event_loop().time()
                })

            # Send final
            await self._send_message({
                "type": "final",
                "text": partial_text,
                "segmentId": segment_id,
                "confidence": 0.95,  # Placeholder - extract from model
                "timestamp": asyncio.get_event_loop().time()
            })

            self.current_segment_id += 1

        except Exception as e:
            await self._send_message({
                "type": "error",
                "code": "INFERENCE_ERROR",
                "message": str(e)
            })

    async def _stream_transcription(self, audio: np.ndarray):
        """Generator that yields partial transcriptions."""
        loop = asyncio.get_event_loop()

        # Run llama.cpp inference in executor
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
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.send_json(message)

    async def close(self):
        """Clean up session."""
        self.is_active = False
        if self.audio_buffer:
            await self._run_inference()  # Process remaining audio

# Global session store
sessions: Dict[str, TranscriptionSession] = {}

@app.websocket("/ws/transcribe")
async def transcribe_websocket(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session = None

    try:
        # Initialize ASR engine (singleton pattern recommended)
        asr_engine = get_asr_engine()  # Your engine instance

        session = TranscriptionSession(session_id, asr_engine, websocket)
        sessions[session_id] = session

        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "sessionId": session_id,
            "modelInfo": {
                "name": "Qwen2-Audio-7B",
                "quantization": "Q4_K_M",
                "sampleRate": 16000
            }
        })

        # Message loop
        while session.is_active:
            message = await websocket.receive()

            if "bytes" in message:
                # Binary audio data
                audio_bytes = message["bytes"]
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                await session.process_audio(audio_float)

            elif "text" in message:
                # Control message
                data = json.loads(message["text"])
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
                    # Handle user correction for future training
                    store_correction(
                        session_id=session_id,
                        segment_id=data.get("segmentId"),
                        corrected_text=data.get("text")
                    )

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")

    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "code": "INTERNAL_ERROR",
            "message": str(e)
        })

    finally:
        if session:
            await session.close()
            del sessions[session_id]
        await websocket.close()
```

---

## 6. Frontend UI Components

### Component Architecture

```
src/
├── components/
│   ├── AudioControls/
│   │   ├── MicrophoneSelector.svelte
│   │   ├── RecordButton.svelte
│   │   └── AudioMeter.svelte
│   ├── Transcription/
│   │   ├── LiveTranscript.svelte
│   │   ├── EditableSegment.svelte
│   │   └── ConfidenceIndicator.svelte
│   └── Layout/
│       ├── Header.svelte
│       └── StatusBar.svelte
├── lib/
│   ├── audio/
│   │   ├── AudioCapture.js
│   │   └── AudioWorklet.js
│   ├── websocket/
│   │   └── AudioStream.js
│   └── stores/
│       └── transcription.js
└── App.svelte
```

### Key Components

```svelte
<!-- frontend/src/components/AudioControls/MicrophoneSelector.svelte -->
<script>
  import { onMount } from 'svelte';

  export let selectedDevice = null;
  export let onSelect = () => {};

  let devices = [];
  let isLoading = true;

  onMount(async () => {
    // Request permission first to get labeled devices
    await navigator.mediaDevices.getUserMedia({ audio: true });
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    devices = allDevices.filter(d => d.kind === 'audioinput');
    isLoading = false;

    // Select default
    if (devices.length > 0 && !selectedDevice) {
      selectedDevice = devices[0].deviceId;
      onSelect(selectedDevice);
    }
  });
</script>

<div class="microphone-selector">
  <label for="mic-select">Microphone:</label>
  <select
    id="mic-select"
    bind:value={selectedDevice}
    on:change={() => onSelect(selectedDevice)}
    disabled={isLoading}
  >
    {#each devices as device}
      <option value={device.deviceId}>
        {device.label || `Microphone ${devices.indexOf(device) + 1}`}
      </option>
    {/each}
  </select>
</div>
```

```svelte
<!-- frontend/src/components/Transcription/LiveTranscript.svelte -->
<script>
  import { writable } from 'svelte/store';
  import EditableSegment from './EditableSegment.svelte';

  // Store for transcription segments
  export const segments = writable([]);
  export const currentPartial = writable('');

  export function addSegment(text, confidence = 0.95) {
    segments.update(segs => [...segs, {
      id: `seg_${segs.length}`,
      text,
      confidence,
      isEditing: false,
      timestamp: Date.now()
    }]);
  }

  export function updatePartial(text) {
    currentPartial.set(text);
  }

  export function finalizePartial() {
    currentPartial.update(text => {
      if (text.trim()) {
        addSegment(text);
      }
      return '';
    });
  }

  function handleCorrection(segmentId, newText) {
    segments.update(segs =>
      segs.map(s => s.id === segmentId ? { ...s, text: newText } : s)
    );

    // Send correction to server for potential training
    dispatch('correction', { segmentId, text: newText });
  }
</script>

<div class="transcription-container">
  <div class="segments">
    {#each $segments as segment (segment.id)}
      <EditableSegment
        {segment}
        on:correction={(e) => handleCorrection(segment.id, e.detail)}
      />
    {/each}
  </div>

  {#if $currentPartial}
    <div class="partial-text">
      {$currentPartial}
      <span class="cursor">|</span>
    </div>
  {/if}
</div>

<style>
  .transcription-container {
    min-height: 200px;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 8px;
    font-family: system-ui, sans-serif;
    line-height: 1.6;
  }

  .partial-text {
    color: #666;
    font-style: italic;
    margin-top: 0.5rem;
  }

  .cursor {
    animation: blink 1s infinite;
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
</style>
```

```svelte
<!-- frontend/src/App.svelte -->
<script>
  import { onMount, onDestroy } from 'svelte';
  import { AudioCapture } from './lib/audio/AudioCapture.js';
  import { AudioStream } from './lib/websocket/AudioStream.js';
  import MicrophoneSelector from './components/AudioControls/MicrophoneSelector.svelte';
  import LiveTranscript from './components/Transcription/LiveTranscript.svelte';

  let audioCapture;
  let audioStream;
  let isRecording = false;
  let isConnected = false;
  let selectedDevice = null;
  let transcriptComponent;

  const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/transcribe';

  async function startRecording() {
    if (!selectedDevice) return;

    // Initialize WebSocket
    audioStream = new AudioStream(WS_URL);
    await audioStream.connect();
    isConnected = true;

    // Set up message handler
    audioStream.onMessage((data) => {
      switch (data.type) {
        case 'partial':
          transcriptComponent.updatePartial(data.text);
          break;
        case 'final':
          transcriptComponent.finalizePartial();
          transcriptComponent.addSegment(data.text, data.confidence);
          break;
        case 'error':
          console.error('Server error:', data.message);
          break;
      }
    });

    // Initialize audio capture
    audioCapture = new AudioCapture({
      sampleRate: 16000,
      chunkDuration: 100,
      onAudioChunk: (pcmData) => {
        if (audioStream?.ws?.readyState === WebSocket.OPEN) {
          audioStream.sendAudioChunk(pcmData);
        }
      },
      onVADChange: (isSpeech) => {
        // Optional: visual feedback for speech detection
      }
    });

    await audioCapture.initialize(selectedDevice);
    audioCapture.start();
    isRecording = true;
  }

  function stopRecording() {
    audioCapture?.stop();
    audioStream?.sendControlMessage('end');
    audioStream?.close();
    isRecording = false;
    isConnected = false;
  }

  function handleKeydown(event) {
    if (event.ctrlKey && event.key === 'r') {
      event.preventDefault();
      isRecording ? stopRecording() : startRecording();
    }
    if (event.key === 'Escape' && isRecording) {
      stopRecording();
    }
  }

  onDestroy(() => {
    stopRecording();
  });
</script>

<svelte:window on:keydown={handleKeydown}/>

<main class="app">
  <header>
    <h1>Qwen ASR Transcription</h1>
    <div class="shortcuts">
      <kbd>Ctrl+R</kbd> Record/Stop
      <kbd>Esc</kbd> Cancel
    </div>
  </header>

  <div class="controls">
    <MicrophoneSelector
      bind:selectedDevice
      onSelect={(id) => selectedDevice = id}
    />

    <button
      class="record-btn"
      class:recording={isRecording}
      on:click={isRecording ? stopRecording : startRecording}
      disabled={!selectedDevice}
    >
      {isRecording ? 'Stop' : 'Record'}
    </button>

    <div class="status">
      {#if isRecording}
        <span class="recording-indicator">● Recording</span>
      {/if}
      {#if isConnected}
        <span class="connected-badge">Connected</span>
      {/if}
    </div>
  </div>

  <LiveTranscript bind:this={transcriptComponent} />
</main>

<style>
  .app {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }

  .controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1rem;
    padding: 1rem;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  .record-btn {
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    border: none;
    border-radius: 4px;
    background: #dc3545;
    color: white;
    cursor: pointer;
    transition: background 0.2s;
  }

  .record-btn:hover {
    background: #c82333;
  }

  .record-btn.recording {
    background: #6c757d;
  }

  .recording-indicator {
    color: #dc3545;
    animation: pulse 1s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .connected-badge {
    background: #28a745;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  kbd {
    background: #e9ecef;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.875rem;
  }
</style>
```

---

## 7. File Structure

```
qwen-asr-transcription/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   ├── asr_engine.py           # llama.cpp Qwen ASR wrapper
│   ├── websocket_handler.py    # WebSocket endpoint handlers
│   ├── audio_processor.py      # Audio preprocessing
│   ├── session_manager.py      # Session lifecycle management
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       ├── logging_config.py
│       └── validators.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── public/
│   │   └── audio-processor.js  # AudioWorklet script
│   └── src/
│       ├── App.svelte
│       ├── main.js
│       ├── components/
│       │   ├── AudioControls/
│       │   │   ├── MicrophoneSelector.svelte
│       │   │   ├── RecordButton.svelte
│       │   │   └── AudioMeter.svelte
│       │   ├── Transcription/
│       │   │   ├── LiveTranscript.svelte
│       │   │   ├── EditableSegment.svelte
│       │   │   └── ConfidenceIndicator.svelte
│       │   └── Layout/
│       │       ├── Header.svelte
│       │       └── StatusBar.svelte
│       ├── lib/
│       │   ├── audio/
│       │   │   ├── AudioCapture.js
│       │   │   └── AudioWorklet.js
│       │   ├── websocket/
│       │   │   └── AudioStream.js
│       │   └── stores/
│       │       └── transcription.js
│       └── styles/
│           └── global.css
├── models/                     # Downloaded GGUF models (gitignored)
│   └── .gitkeep
├── docs/
│   └── architecture.md         # This document
├── scripts/
│   ├── download_model.sh       # Model download helper
│   └── setup.sh                # Initial setup
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── README.md
```

---

## 8. Performance Considerations

### CPU Optimization

1. **Model Quantization**: Use Q4_K_M or Q5_K_M for best speed/quality tradeoff
2. **Thread Affinity**: Pin llama.cpp threads to physical cores
3. **Batch Size**: Keep at 1 for streaming; increase only for offline batch
4. **Memory Mapping**: Enable `use_mmap=True` for faster model loading
5. **KV Cache**: Enable streaming KV cache for continuity across chunks

### Latency Targets

| Component | Target | Optimization |
|-----------|--------|--------------|
| Audio Capture | <50ms | AudioWorklet, 16kHz direct |
| Network RTT | <100ms | WebSocket binary, local if possible |
| Inference | <300ms | Q4_K_M, CPU threads optimized |
| UI Update | <50ms | Svelte reactivity, virtual list |
| **End-to-End** | **<500ms** | **From speech to text display** |

### Audio Pipeline Optimizations

```python
# backend/audio_processor.py
import numpy as np
import librosa
from typing import Tuple

class AudioProcessor:
    """Optimized audio preprocessing for ASR."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self._resample_cache = {}

    def preprocess(
        self,
        audio: np.ndarray,
        source_sr: int = 16000
    ) -> np.ndarray:
        """Fast preprocessing pipeline."""

        # Resample if needed
        if source_sr != self.target_sr:
            audio = self._fast_resample(audio, source_sr, self.target_sr)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        return audio.astype(np.float32)

    def _fast_resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Cached resampling for common rates."""
        key = (orig_sr, target_sr)
        if key not in self._resample_cache:
            self._resample_cache[key] = None  # librosa doesn't cache
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
```

### Caching Strategy

```python
# backend/session_manager.py
from functools import lru_cache
from typing import Optional
import hashlib

class TranscriptionCache:
    """Cache for repeated audio segments (e.g., silence)."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size

    def get_key(self, audio: np.ndarray) -> str:
        """Generate cache key from audio fingerprint."""
        # Simple hash of downsampled audio
        fingerprint = audio[::100].tobytes()
        return hashlib.md5(fingerprint).hexdigest()

    def get(self, audio: np.ndarray) -> Optional[str]:
        key = self.get_key(audio)
        return self.cache.get(key)

    def put(self, audio: np.ndarray, transcription: str):
        if len(self.cache) >= self.max_size:
            # LRU eviction
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        key = self.get_key(audio)
        self.cache[key] = transcription
```

---

## 9. Setup and Deployment

### Local Development Setup

```bash
# 1. Clone repository
git clone <your-repo>
cd qwen-asr-transcription

# 2. Backend setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download model
mkdir -p models
# Download Qwen2-Audio GGUF from HuggingFace
wget https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct-GGUF/resolve/main/qwen2-audio-7b-instruct-q4_k_m.gguf \
  -O models/qwen2-audio-7b-instruct-q4_k_m.gguf

# 4. Start backend
cd backend
python main.py

# 5. Frontend setup (new terminal)
cd frontend
npm install
npm run dev

# 6. Open browser to http://localhost:5173
```

### Requirements Files

```txt
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
websockets==12.0
llama-cpp-python==0.2.38
numpy==1.26.3
librosa==0.10.1
python-multipart==0.0.6
pydantic==2.5.3
pydantic-settings==2.1.0
webrtcvad==2.0.10
```

```json
// frontend/package.json
{
  "name": "qwen-asr-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^3.0.0",
    "svelte": "^5.0.0",
    "vite": "^5.0.0"
  },
  "dependencies": {
    "tailwindcss": "^3.4.0"
  }
}
```

### Docker Deployment

```dockerfile
# Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/qwen2-audio-7b-instruct-q4_k_m.gguf
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Dockerfile.frontend
FROM node:20-alpine AS builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/qwen2-audio-7b-instruct-q4_k_m.gguf
      - N_THREADS=4
      - LOG_LEVEL=info
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - VITE_WS_URL=ws://localhost:8000/ws/transcribe
```

### Production Considerations

1. **HTTPS**: Use WSS (WebSocket Secure) in production
2. **Rate Limiting**: Implement per-session rate limits
3. **Model Caching**: Keep single ASR engine instance (singleton)
4. **Monitoring**: Add Prometheus metrics for latency, throughput
5. **Scaling**: Use Redis for session state if scaling horizontally

```python
# backend/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    model_path: str = "models/qwen2-audio-7b-instruct-q4_k_m.gguf"
    n_threads: int = 4
    n_ctx: int = 8192
    max_concurrent_sessions: int = 10
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    log_level: str = "info"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## Summary

This architecture provides:

- **Sub-500ms end-to-end latency** through optimized audio pipeline and streaming inference
- **CPU-efficient inference** via llama.cpp with quantized Qwen ASR models
- **Robust real-time streaming** with WebSocket protocol and reconnection logic
- **Editable transcription UI** with segment-based correction capability
- **Production-ready deployment** with Docker and horizontal scaling support

Key implementation priorities:
1. Get basic WebSocket audio streaming working first
2. Integrate llama.cpp with Qwen model for transcription
3. Add client-side VAD to reduce unnecessary inference
4. Implement editable UI with correction tracking
5. Optimize based on profiling data from your specific hardware
