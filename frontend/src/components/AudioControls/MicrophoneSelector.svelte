<script>
  import { onMount } from 'svelte';

  export let selectedDevice = null;
  export let onSelect = () => {};

  let devices = [];
  let isLoading = true;
  let error = null;

  onMount(async () => {
    try {
      // Request permission first to get labeled devices
      await navigator.mediaDevices.getUserMedia({ audio: true });
      await loadDevices();

      // Listen for device changes
      navigator.mediaDevices.addEventListener('devicechange', loadDevices);

      return () => {
        navigator.mediaDevices.removeEventListener('devicechange', loadDevices);
      };
    } catch (err) {
      error = 'Microphone permission denied';
      isLoading = false;
    }
  });

  async function loadDevices() {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      devices = allDevices.filter(d => d.kind === 'audioinput');
      isLoading = false;

      // Select default if none selected
      if (devices.length > 0 && !selectedDevice) {
        selectedDevice = devices[0].deviceId;
        onSelect(selectedDevice);
      }
    } catch (err) {
      error = 'Failed to enumerate devices';
      isLoading = false;
    }
  }

  function handleChange(event) {
    selectedDevice = event.target.value;
    onSelect(selectedDevice);
  }
</script>

<div class="microphone-selector">
  <label for="mic-select">Microphone</label>
  {#if error}
    <span class="error">{error}</span>
  {:else}
    <select
      id="mic-select"
      value={selectedDevice}
      on:change={handleChange}
      disabled={isLoading || devices.length === 0}
    >
      {#if isLoading}
        <option>Loading...</option>
      {:else if devices.length === 0}
        <option>No microphones found</option>
      {:else}
        {#each devices as device}
          <option value={device.deviceId}>
            {device.label || `Microphone ${devices.indexOf(device) + 1}`}
          </option>
        {/each}
      {/if}
    </select>
  {/if}
</div>

<style>
  .microphone-selector {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #555;
  }

  select {
    padding: 0.5rem;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    font-size: 0.9rem;
    min-width: 200px;
    background: white;
  }

  select:focus {
    outline: none;
    border-color: #4a90d9;
    box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.2);
  }

  select:disabled {
    background: #f5f5f5;
    cursor: not-allowed;
  }

  .error {
    color: #dc3545;
    font-size: 0.8rem;
  }
</style>
