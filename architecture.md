# Qwen ASR Transcription System Architecture

## Overview

A high-performance web application for real-time speech transcription using Qwen ASR models with OpenVINO inference. This system provides low-latency transcription with support for both microphone input and audio file uploads.

## System Components

### 1. Frontend

The frontend is a Svelte-based web application that provides a user-friendly interface for interacting with the transcription system.

#### Key Components:

- **App.svelte**: Main application component that manages state and coordination between components
- **MicrophoneSelector.svelte**: Component for selecting audio input devices
- **FileUploader.svelte**: Component for uploading audio files
- **LiveTranscript.svelte**: Component for displaying real-time transcription results
- **AudioCapture.js**: Utility for capturing audio from the microphone
- **AudioStream.js**: WebSocket client for streaming audio to the backend

#### Features:
- Real-time audio visualization
- Microphone selection
- Audio file upload and preview
- Live transcription display
- Editable transcript segments
- Keyboard shortcuts for control

### 2. Backend

The backend is a FastAPI application that handles WebSocket connections, audio processing, and model inference.

#### Key Components:

- **main.py**: FastAPI application with WebSocket and HTTP endpoints
- **asr_engine.py**: OpenVINO-based ASR inference engine
- **audio_processor.py**: Audio preprocessing utilities
- **config.py**: Configuration management

#### Endpoints:
- **WebSocket /ws/transcribe**: Real-time audio streaming and transcription
- **HTTP POST /api/transcribe/file**: File-based transcription
- **HTTP POST /api/transcribe/audio**: Raw audio data transcription
- **HTTP GET /health**: Health check
- **HTTP GET /model-info**: Model information

### 3. Inference Engine

The inference engine uses OpenVINO for optimized model inference on CPU, GPU, or NPU.

#### Key Components:

- **QwenASREngine**: Wrapper around the OpenVINO ASR model
- **OVQwen3ASRModel**: OpenVINO-optimized Qwen3-ASR model
- **Qwen3-ASR**: Original model implementation

#### Features:
- OpenVINO-optimized inference
- Support for multiple devices (CPU, GPU, NPU)
- Batch processing for improved performance
- Streaming transcription with partial results
- Language auto-detection

## Data Flow

### Microphone Input Flow

1. **Audio Capture**: The frontend captures audio from the selected microphone using the Web Audio API
2. **VAD Processing**: Client-side Voice Activity Detection reduces unnecessary data transfer
3. **WebSocket Streaming**: Audio chunks are streamed to the backend via WebSocket
4. **Audio Processing**: Backend processes and normalizes the audio data
5. **Model Inference**: OpenVINO model transcribes the audio
6. **Result Streaming**: Transcription results are streamed back to the frontend
7. **Display**: Frontend displays partial and final transcription results

### File Upload Flow

1. **File Selection**: User selects an audio file via the frontend
2. **File Preview**: Frontend provides audio preview functionality
3. **File Processing**: Frontend reads and decodes the audio file
4. **Chunk Streaming**: Audio is streamed to the backend in chunks via WebSocket
5. **Model Inference**: OpenVINO model transcribes the audio
6. **Result Display**: Transcription results are displayed as they become available

## Technical Stack

### Frontend
- **Framework**: Svelte 5
- **Build Tool**: Vite
- **Web API**: Web Audio API, WebSocket API
- **Styling**: CSS3

### Backend
- **Framework**: FastAPI
- **WebSockets**: FastAPI WebSocket support
- **Audio Processing**: NumPy, optional Librosa
- **Inference**: OpenVINO Runtime
- **Model**: Qwen3-ASR-1.7B (OpenVINO format)

### Infrastructure
- **Deployment**: Docker Compose
- **Networking**: WebSockets for real-time communication
- **Storage**: Temporary file storage for uploaded audio files

## Performance Optimization

### Frontend Optimization
- **AudioWorklet**: Efficient audio processing in a separate thread
- **Chunking**: Small audio chunks for low latency
- **VAD**: Client-side voice activity detection to reduce data transfer

### Backend Optimization
- **OpenVINO**: Hardware-optimized inference
- **Batch Processing**: Parallel inference for multiple audio chunks
- **Async Processing**: Non-blocking WebSocket handling
- **Buffer Management**: Efficient audio buffer management

### Model Optimization
- **OpenVINO Conversion**: Model optimized for inference
- **Quantization**: Reduced precision for faster inference
- **Device Selection**: Automatic device selection based on availability

## Configuration

### Backend Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `../converted_models/Qwen3-ASR-1.7B-OV` | Path to OpenVINO model |
| `OV_DEVICE` | `CPU` | OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `OV_MAX_BATCH_SIZE` | `32` | Maximum inference batch size |
| `OV_MAX_NEW_TOKENS` | `512` | Maximum tokens to generate |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level |
| `MAX_BUFFER_MS` | `5000` | Audio buffer duration in milliseconds |

### Frontend Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VITE_WS_URL` | `ws://localhost:8000/ws/transcribe` | WebSocket URL |
| `sampleRate` | `16000` | Audio sample rate |
| `chunkDuration` | `100` | Audio chunk duration in milliseconds |
| `energyThreshold` | `0.01` | VAD energy threshold |

## Deployment

### Local Development

```bash
# Start backend
cd backend
python main.py

# Start frontend
cd frontend
npm run dev
```

### Docker Deployment

```bash
docker-compose up --build
```

## Security Considerations

- **Audio Data**: Audio data is transmitted over WebSockets (unencrypted in development)
- **File Uploads**: Uploaded files are stored temporarily and deleted after processing
- **API Access**: No authentication is implemented (for development purposes)
- **CORS**: CORS is enabled for all origins (for development purposes)

## Future Enhancements

- **Authentication**: Add user authentication for secure access
- **Encryption**: Implement WebSocket encryption (WSS)
- **Model Selection**: Support for multiple ASR models
- **Language Support**: Enhanced language detection and support
- **Speaker Diarization**: Speaker separation in multi-speaker scenarios
- **Custom Vocabulary**: Domain-specific vocabulary support
- **Real-time Translation**: Integrated translation of transcriptions
- **Cloud Deployment**: Scalable cloud deployment options

## Troubleshooting

### Common Issues

1. **Microphone Access Denied**: Ensure the browser has permission to access the microphone
2. **WebSocket Connection Failed**: Ensure the backend server is running
3. **Transcription Accuracy**: Check audio quality and background noise levels
4. **Performance Issues**: Adjust batch size and buffer duration based on hardware

### Debugging

- **Frontend**: Use browser developer tools to inspect WebSocket traffic and console logs
- **Backend**: Check server logs for error messages and performance metrics
- **Model**: Verify OpenVINO model conversion and device compatibility

## Conclusion

The Qwen ASR Transcription System provides a high-performance, low-latency solution for real-time speech transcription. By leveraging OpenVINO for optimized inference and WebSockets for real-time communication, the system delivers accurate transcriptions with minimal delay. The architecture is modular and extensible, allowing for future enhancements and integration with other systems.