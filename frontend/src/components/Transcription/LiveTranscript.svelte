<script>
  import { writable } from 'svelte/store';
  import EditableSegment from './EditableSegment.svelte';

  // Store for transcription segments
  const segments = writable([]);
  const currentPartial = writable('');

  export function addSegment(text, confidence = 0.95) {
    if (!text || !text.trim()) return;

    segments.update(segs => [...segs, {
      id: `seg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      text: text.trim(),
      confidence,
      timestamp: Date.now()
    }]);
  }

  export function updatePartial(text) {
    currentPartial.set(text);
  }

  export function clearPartial() {
    currentPartial.set('');
  }

  function handleCorrection(segmentId, newText) {
    segments.update(segs =>
      segs.map(s => s.id === segmentId ? { ...s, text: newText } : s)
    );

    // Dispatch event for parent to send to server
    const event = new CustomEvent('correction', {
      detail: { segmentId, text: newText },
      bubbles: true
    });
    dispatchEvent(event);
  }

  function clearAll() {
    segments.set([]);
    currentPartial.set('');
  }

  function copyToClipboard() {
    const fullText = $segments.map(s => s.text).join(' ');
    navigator.clipboard.writeText(fullText);
  }
</script>

<div class="transcription-container">
  <div class="toolbar">
    <span class="segment-count">{$segments.length} segments</span>
    <div class="actions">
      <button class="action-btn" on:click={copyToClipboard} disabled={$segments.length === 0}>
        Copy All
      </button>
      <button class="action-btn danger" on:click={clearAll} disabled={$segments.length === 0}>
        Clear
      </button>
    </div>
  </div>

  <div class="transcript-area">
    {#if $segments.length === 0 && !$currentPartial}
      <div class="placeholder">
        Transcription will appear here...
      </div>
    {:else}
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
    {/if}
  </div>
</div>

<style>
  .transcription-container {
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    overflow: hidden;
  }

  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: #f8f9fa;
    border-bottom: 1px solid #e0e0e0;
  }

  .segment-count {
    font-size: 0.8rem;
    color: #666;
  }

  .actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-btn {
    padding: 0.375rem 0.75rem;
    font-size: 0.8rem;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    transition: all 0.2s;
  }

  .action-btn:hover:not(:disabled) {
    background: #f0f0f0;
  }

  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .action-btn.danger {
    color: #dc3545;
    border-color: #dc3545;
  }

  .action-btn.danger:hover:not(:disabled) {
    background: #fee;
  }

  .transcript-area {
    min-height: 300px;
    max-height: 500px;
    overflow-y: auto;
    padding: 1rem;
  }

  .placeholder {
    color: #999;
    text-align: center;
    padding: 3rem;
    font-style: italic;
  }

  .segments {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    align-items: baseline;
  }

  .partial-text {
    color: #666;
    font-style: italic;
    margin-top: 0.5rem;
    display: inline;
  }

  .cursor {
    animation: blink 1s infinite;
    color: #4a90d9;
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
</style>
