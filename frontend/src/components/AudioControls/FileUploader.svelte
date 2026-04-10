<script>
  import { onMount } from 'svelte';

  export let selectedFile = null;
  export let onFileSelect = () => {};
  export let onFileUpload = () => {};
  export let isUploading = false;
  
  let fileInput;
  let fileError = null;
  let audioElement;
  
  function handleFileChange(event) {
    const file = event.target.files[0];
    if (!file) {
      selectedFile = null;
      fileError = null;
      onFileSelect(null);
      return;
    }
    
    // Validate file type
    if (!file.type.startsWith('audio/')) {
      fileError = 'Please select an audio file';
      selectedFile = null;
      onFileSelect(null);
      return;
    }
    
    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      fileError = 'File size exceeds 100MB limit';
      selectedFile = null;
      onFileSelect(null);
      return;
    }
    
    // Preview the file
    const url = URL.createObjectURL(file);
    if (audioElement) {
      audioElement.src = url;
    }
    
    selectedFile = file;
    fileError = null;
    onFileSelect(file);
  }
  
  function clearFile() {
    selectedFile = null;
    fileError = null;
    if (fileInput) {
      fileInput.value = '';
    }
    if (audioElement) {
      audioElement.src = '';
    }
    onFileSelect(null);
  }
  
  function uploadFile() {
    if (selectedFile) {
      onFileUpload(selectedFile);
    }
  }
</script>

<div class="file-uploader">
  <label for="file-input">Audio File</label>
  
  <div class="file-input-container">
    <input
      type="file"
      id="file-input"
      accept="audio/*"
      bind:this={fileInput}
      on:change={handleFileChange}
      disabled={isUploading}
    />
    
    {#if selectedFile}
      <div class="file-info">
        <span class="file-name">{selectedFile.name}</span>
        <span class="file-size">{Math.round(selectedFile.size / 1024)} KB</span>
        <button class="clear-btn" on:click={clearFile} title="Clear file">×</button>
      </div>
      
      <audio 
        bind:this={audioElement}
        controls 
        class="audio-preview"
        style="width: 100%; margin-top: 0.5rem;"
      >
        Your browser does not support the audio element.
      </audio>
      
      <button 
        class="upload-btn"
        on:click={uploadFile}
        disabled={isUploading}
      >
        {isUploading ? 'Uploading...' : 'Upload and Transcribe'}
      </button>
    {/if}
    
    {#if fileError}
      <span class="error">{fileError}</span>
    {/if}
  </div>
</div>

<style>
  .file-uploader {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #555;
  }
  
  .file-input-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  input[type="file"] {
    padding: 0.5rem;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    font-size: 0.9rem;
    background: white;
  }
  
  .file-info {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem;
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    font-size: 0.8rem;
  }
  
  .file-name {
    flex: 1;
    font-weight: 500;
  }
  
  .file-size {
    color: #6c757d;
    margin-left: 0.5rem;
  }
  
  .clear-btn {
    background: none;
    border: none;
    font-size: 1.2rem;
    cursor: pointer;
    color: #6c757d;
    padding: 0 0.25rem;
  }
  
  .clear-btn:hover {
    color: #dc3545;
  }
  
  .audio-preview {
    border: 1px solid #e9ecef;
    border-radius: 4px;
  }
  
  .upload-btn {
    padding: 0.5rem 1rem;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: background 0.2s;
  }
  
  .upload-btn:hover {
    background: #357abd;
  }
  
  .upload-btn:disabled {
    background: #6c757d;
    cursor: not-allowed;
  }
  
  .error {
    color: #dc3545;
    font-size: 0.8rem;
  }
</style>