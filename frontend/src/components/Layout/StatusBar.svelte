<script>
  export let isConnected = false;
  export let isRecording = false;
  export let latency = null;
  export let modelName = "";
</script>

<div class="status-bar">
  <div class="status-group">
    {#if modelName}
      <span class="status-item">
        Model: <strong>{modelName}</strong>
      </span>
    {/if}
    <span class="status-item" class:active={isConnected}>
      {isConnected ? '● Connected' : '○ Disconnected'}
    </span>
    {#if isRecording}
      <span class="status-item recording">
        ● Recording
      </span>
    {/if}
  </div>

  {#if latency !== null}
    <div class="latency">
      Latency: {latency}ms
    </div>
  {/if}
</div>

<style>
  .status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 1rem;
    background: #f8f9fa;
    border-top: 1px solid #e0e0e0;
    font-size: 0.8rem;
    color: #666;
  }

  .status-group {
    display: flex;
    gap: 1rem;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .status-item.active {
    color: #28a745;
  }

  .status-item.recording {
    color: #dc3545;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .latency {
    font-family: ui-monospace, monospace;
  }
</style>
