# Qwen ASR + llama.cpp Real-Time Transcription System

A high-performance web application for real-time speech transcription using Qwen ASR models with llama.cpp CPU inference. With sophisticated algorithms for smart transcription and correction.

## Features

- **Real-time transcription** with sub-500ms latency
- **CPU-optimized inference** using OpenVINO with Qwen3-ASR models
- **Web-based interface** with microphone selection and live editing
- **WebSocket streaming** for bidirectional audio/text communication
- **Client-side VAD** to reduce unnecessary inference
- **Editable transcriptions** with correction tracking

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 20+
- 4GB+ RAM (for 1.7B model)
- Microphone access

### Installation

```bash
# Clone repository
git clone <repo-url>
cd qwen-asr-transcription

# Setup backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup Qwen3-ASR package (required for OpenVINO inference)
cd get_convert_qwenasr/Qwen3-ASR
pip install -e .
cd ../..

# Convert model to OpenVINO format (or download pre-converted)
mkdir -p converted_models
# Option 1: Convert from HuggingFace
python convert_to_ov.py --model Qwen/Qwen3-ASR-1.7B --output converted_models/Qwen3-ASR-1.7B-OV

# Option 2: Download pre-converted model
# wget <pre-converted-model-url> -O converted_models/Qwen3-ASR-1.7B-OV.zip

# Setup frontend
cd frontend
npm install
```

### Running

```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev

# Open http://localhost:5173
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for complete system design.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Start/Stop recording |
| `Esc` | Cancel recording |

## Configuration

The backend can be configured using environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `../converted_models/Qwen3-ASR-1.7B-OV` | Path to OpenVINO model |
| `OV_DEVICE` | `CPU` | OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `OV_MAX_BATCH_SIZE` | `32` | Maximum inference batch size |
| `OV_MAX_NEW_TOKENS` | `512` | Maximum tokens to generate |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level |

## Architecture

The system uses:
- **OpenVINO Runtime** for optimized model inference on CPU/GPU/NPU
- **Qwen3-ASR-1.7B** model converted to OpenVINO format
- **WebSocket streaming** for real-time audio processing with ~500ms latency

## Docker Deployment

```bash
docker-compose up --build
```

## License

MIT
