<script>
  export let title = "Qwen ASR Transcription";
</script>

<header>
  <div class="logo">
    <h1>{title}</h1>
  </div>
  <nav>
    <slot></slot>
  </nav>
</header>

<style>
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: #fff;
    border-bottom: 1px solid #e0e0e0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  .logo h1 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
  }

  nav {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
</style>
