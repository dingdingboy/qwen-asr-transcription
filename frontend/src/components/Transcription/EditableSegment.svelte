<script>
  import { createEventDispatcher } from 'svelte';

  export let segment;

  const dispatch = createEventDispatcher();

  let isEditing = false;
  let editText = segment.text;
  let editInput;

  function startEdit() {
    isEditing = true;
    editText = segment.text;
    // Focus input after render
    setTimeout(() => editInput?.focus(), 0);
  }

  function saveEdit() {
    if (editText.trim() !== segment.text) {
      dispatch('correction', editText.trim());
    }
    isEditing = false;
  }

  function cancelEdit() {
    editText = segment.text;
    isEditing = false;
  }

  function handleKeydown(event) {
    if (event.key === 'Enter') {
      saveEdit();
    } else if (event.key === 'Escape') {
      cancelEdit();
    }
  }

  function getConfidenceColor(confidence) {
    if (confidence >= 0.9) return '#28a745';
    if (confidence >= 0.7) return '#ffc107';
    return '#dc3545';
  }
</script>

{#if isEditing}
  <span class="segment editing">
    <input
      bind:this={editInput}
      bind:value={editText}
      on:keydown={handleKeydown}
      on:blur={saveEdit}
      class="edit-input"
    />
  </span>
{:else}
  <span
    class="segment"
     on:dblclick={startEdit}
     title="Double-click to edit"
     style="--confidence-color: {getConfidenceColor(segment.confidence)}"
     role="button"
     tabindex="0"
  >
    {segment.text}
  </span>
{/if}

<style>
  .segment {
    display: inline;
    padding: 0.125rem 0.25rem;
    margin: 0.125rem 0;
    border-radius: 3px;
    cursor: pointer;
    transition: background 0.2s;
    border-bottom: 2px solid var(--confidence-color, transparent);
  }

  .segment:hover {
    background: #f0f0f0;
  }

  .segment.editing {
    padding: 0;
  }

  .edit-input {
    font: inherit;
    padding: 0.125rem 0.25rem;
    border: 2px solid #4a90d9;
    border-radius: 3px;
    outline: none;
    min-width: 100px;
    background: white;
  }
</style>
