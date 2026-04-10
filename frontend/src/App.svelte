<script>
  import { onMount, onDestroy } from 'svelte';
  import MicrophoneSelector from './components/AudioControls/MicrophoneSelector.svelte';
  import FileUploader from './components/AudioControls/FileUploader.svelte';
  import LiveTranscript from './components/Transcription/LiveTranscript.svelte';
  import { AudioCapture } from './lib/audio/AudioCapture.js';
  import { AudioStream } from './lib/websocket/AudioStream.js';

  let audioCapture = null;
  let audioStream = null;
  let isRecording = false;
  let isConnected = false;
  let isConnecting = false;
  let selectedDevice = null;
  let selectedFile = null;
  let transcriptComponent;
  let errorMessage = '';
  let isSpeechDetected = false;
  let isUploading = false;
  let inputMode = 'microphone'; // 'microphone' or 'file'

  const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/transcribe';

  async function startRecording() {
    if (!selectedDevice) {
      errorMessage = 'Please select a microphone first';
      return;
    }

    errorMessage = '';
    isConnecting = true;

    try {
      // Initialize WebSocket
      audioStream = new AudioStream(WS_URL);
      await audioStream.connect();
      isConnected = true;
      isConnecting = false;

      // Set up message handler
      audioStream.onMessage((data) => {
        handleStreamMessage(data);
      });

      // Initialize audio capture
      audioCapture = new AudioCapture({
        sampleRate: 16000,
        chunkDuration: 100,
        onAudioChunk: (pcmData) => {
          if (audioStream?.isOpen()) {
            audioStream.sendAudioChunk(pcmData);
          }
        },
        onVADChange: (isSpeech) => {
          isSpeechDetected = isSpeech;
        }
      });

      await audioCapture.initialize(selectedDevice);
      audioCapture.start();
      isRecording = true;

    } catch (error) {
      console.error('Failed to start recording:', error);
      errorMessage = `Failed to start: ${error.message}`;
      isConnecting = false;
      cleanup();
    }
  }

  async function uploadFile(file) {
    errorMessage = '';
    isUploading = true;

    try {
      // Initialize WebSocket
      audioStream = new AudioStream(WS_URL);
      await audioStream.connect();
      isConnected = true;

      // Set up message handler
      audioStream.onMessage((data) => {
        handleStreamMessage(data);
      });

      // Read and process the audio file
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to float32 array
      const channelData = audioBuffer.getChannelData(0);
      
      // Process in chunks
      const chunkSize = 16000 * 0.1; // 100ms chunks
      let position = 0;
      
      const processChunk = async () => {
        if (position >= channelData.length) {
          // Send end message
          if (audioStream?.isOpen()) {
            audioStream.sendControlMessage('end');
          }
          isUploading = false;
          return;
        }
        
        const end = Math.min(position + chunkSize, channelData.length);
        const chunk = channelData.slice(position, end);
        
        if (audioStream?.isOpen()) {
          audioStream.sendAudioChunk(chunk);
        }
        
        position = end;
        
        // Use setTimeout to avoid blocking the UI
        setTimeout(processChunk, 50);
      };
      
      // Start processing
      processChunk();

    } catch (error) {
      console.error('Failed to upload file:', error);
      errorMessage = `Failed to upload: ${error.message}`;
      isUploading = false;
      cleanup();
    }
  }

  function handleStreamMessage(data) {
    switch (data.type) {
      case 'ready':
        console.log('Session ready:', data.sessionId);
        break;
      case 'partial':
        transcriptComponent?.updatePartial(data.text);
        break;
      case 'final':
        transcriptComponent?.addSegment(data.text, data.confidence);
        transcriptComponent?.clearPartial();
        break;
      case 'error':
        console.error('Server error:', data.message);
        errorMessage = `Server error: ${data.message}`;
        break;
      case 'pong':
        // Latency calculation could go here
        break;
    }
  }

  function stopRecording() {
    cleanup();
    isRecording = false;
    isConnected = false;
    isSpeechDetected = false;
  }

  function cleanup() {
    audioCapture?.stop();
    audioCapture = null;

    if (audioStream?.isOpen()) {
      audioStream.sendControlMessage('end');
      audioStream.close();
    }
    audioStream = null;
  }

  function handleKeydown(event) {
    // Ctrl+R to toggle recording
    if (event.ctrlKey && event.key === 'r' && inputMode === 'microphone') {
      event.preventDefault();
      isRecording ? stopRecording() : startRecording();
    }
    // Escape to stop
    if (event.key === 'Escape' && (isRecording || isUploading)) {
      stopRecording();
    }
  }

  function handleModeChange(mode) {
    // Stop any ongoing recording/upload
    if (isRecording || isUploading) {
      cleanup();
      isRecording = false;
      isUploading = false;
      isConnected = false;
      isSpeechDetected = false;
    }
    
    inputMode = mode;
    // Clear any existing file selection when switching to microphone
    if (mode === 'microphone') {
      selectedFile = null;
    }
  }

  onDestroy(() => {
    cleanup();
  });
</script>

<svelte:window on:keydown={handleKeydown}/>

<main class="app">
  <header>
    <div class="title-section">
      <h1>Qwen ASR Transcription</h1>
      <p class="subtitle">Real-time speech-to-text with local AI</p>
    </div>
    <div class="shortcuts">
      <kbd>Ctrl</kbd> + <kbd>R</kbd> Record/Stop
      <span class="separator">|</span>
      <kbd>Esc</kbd> Cancel
    </div>
  </header>

  {#if errorMessage}
    <div class="error-banner" role="alert">
      {errorMessage}
      <button class="close-btn" on:click={() => errorMessage = ''}>×</button>
    </div>
  {/if}

  <!-- Input Mode Selector -->
  <div class="mode-selector">
    <button 
      class={`mode-btn ${inputMode === 'microphone' ? 'active' : ''}`}
      on:click={() => handleModeChange('microphone')}
    >
      Microphone
    </button>
    <button 
      class={`mode-btn ${inputMode === 'file' ? 'active' : ''}`}
      on:click={() => handleModeChange('file')}
    >
      Audio File
    </button>
  </div>

  <div class="controls">
    {#if inputMode === 'microphone'}
      <MicrophoneSelector
        bind:selectedDevice
        onSelect={(id) => selectedDevice = id}
      />

      <button
        class="record-btn"
        class:recording={isRecording}
        class:connecting={isConnecting}
        on:click={isRecording ? stopRecording : startRecording}
        disabled={!selectedDevice || isConnecting}
      >
        {#if isConnecting}
          Connecting...
        {:else if isRecording}
          <span class="record-icon">■</span> Stop
        {:else}
          <span class="record-icon">●</span> Record
        {/if}
      </button>

      <div class="status">
        {#if isRecording}
          <span class="recording-indicator" class:active={isSpeechDetected}>
            {isSpeechDetected ? '● Speaking' : '● Listening'}
          </span>
        {/if}
        {#if isConnected}
          <span class="connected-badge">Connected</span>
        {/if}
      </div>
    {:else if inputMode === 'file'}
      <FileUploader
        bind:selectedFile
        onFileSelect={(file) => selectedFile = file}
        onFileUpload={uploadFile}
        isUploading={isUploading}
      />
    {/if}
  </div>

  <LiveTranscript bind:this={transcriptComponent} />
</main>

<style>
  .app {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e0e0e0;
  }

  .title-section h1 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
  }

  .subtitle {
    color: #666;
    font-size: 0.9rem;
    margin: 0.25rem 0 0 0;
  }

  .shortcuts {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: #666;
  }

  .separator {
    color: #ccc;
  }

  .error-banner {
    background: #fee;
    color: #c33;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 1.25rem;
    cursor: pointer;
    color: #c33;
  }

  .mode-selector {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  
  .mode-btn {
    padding: 0.5rem 1rem;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    background: #f8f9fa;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
  }
  
  .mode-btn.active {
    background: #4a90d9;
    color: white;
    border-color: #4a90d9;
  }
  
  .controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  
  @media (min-width: 768px) {
    .controls {
      flex-direction: row;
      align-items: center;
    }
  }

  .record-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.625rem 1.25rem;
    font-size: 0.95rem;
    font-weight: 500;
    border: none;
    border-radius: 6px;
    background: #dc3545;
    color: white;
    cursor: pointer;
    transition: all 0.2s;
  }

  .record-btn:hover:not(:disabled) {
    background: #c82333;
    transform: translateY(-1px);
  }

  .record-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  .record-btn.recording {
    background: #6c757d;
  }

  .record-btn.connecting {
    background: #ffc107;
    color: #333;
  }

  .record-icon {
    font-size: 0.8rem;
  }

  .recording.recording .record-icon {
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .status {
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }

  .recording-indicator {
    color: #dc3545;
    font-size: 0.85rem;
    font-weight: 500;
  }

  .recording-indicator.active {
    color: #28a745;
  }

  .connected-badge {
    background: #28a745;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
  }

  kbd {
    background: #e9ecef;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
    border: 1px solid #d0d0d0;
  }
</style>