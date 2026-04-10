<script>
  export let confidence = 0.95;

  $: color = getConfidenceColor(confidence);
  $: label = getConfidenceLabel(confidence);

  function getConfidenceColor(c) {
    if (c >= 0.9) return '#28a745';
    if (c >= 0.7) return '#ffc107';
    return '#dc3545';
  }

  function getConfidenceLabel(c) {
    if (c >= 0.9) return 'High';
    if (c >= 0.7) return 'Medium';
    return 'Low';
  }
</script>

<span class="confidence-indicator" style="--confidence-color: {color}" title="Confidence: {(confidence * 100).toFixed(0)}%">
  {label}
</span>

<style>
  .confidence-indicator {
    display: inline-block;
    padding: 0.125rem 0.375rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--confidence-color);
    background: color-mix(in srgb, var(--confidence-color) 15%, white);
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.025em;
  }
</style>
