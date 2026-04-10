<script>
  import { onMount, onDestroy } from 'svelte';

  export let audioContext = null;
  export let mediaStream = null;

  let canvas;
  let animationId;
  let analyser;
  let dataArray;

  $: if (audioContext && mediaStream && canvas) {
    setupAnalyser();
  }

  function setupAnalyser() {
    if (!audioContext || !mediaStream) return;

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    const source = audioContext.createMediaStreamSource(mediaStream);
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    draw();
  }

  function draw() {
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    analyser.getByteFrequencyData(dataArray);

    // Clear canvas
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);

    // Calculate average level
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i];
    }
    const average = sum / dataArray.length;
    const level = average / 255;

    // Draw level bar
    const barWidth = width * 0.8;
    const barHeight = height * 0.6;
    const barX = (width - barWidth) / 2;
    const barY = (height - barHeight) / 2;

    // Background
    ctx.fillStyle = '#ddd';
    ctx.fillRect(barX, barY, barWidth, barHeight);

    // Level
    const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
    gradient.addColorStop(0, '#28a745');
    gradient.addColorStop(0.5, '#ffc107');
    gradient.addColorStop(1, '#dc3545');

    ctx.fillStyle = gradient;
    ctx.fillRect(barX, barY, barWidth * level, barHeight);

    // Border
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, barWidth, barHeight);

    animationId = requestAnimationFrame(draw);
  }

  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  });
</script>

<canvas bind:this={canvas} width="100" height="30"></canvas>

<style>
  canvas {
    border-radius: 4px;
    vertical-align: middle;
  }
</style>
