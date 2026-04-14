<script lang="ts">
  import { Upload, Image as ImageIcon } from 'lucide-svelte';

  interface Props {
    onImageSelected: (file: File, dataUrl: string) => void;
    previewUrl?: string | null;
  }

  let { onImageSelected, previewUrl = null }: Props = $props();

  let isDragging = $state(false);
  let fileInput: HTMLInputElement;

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    isDragging = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    isDragging = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    isDragging = false;

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }

  function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const files = input.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }

  function processFile(file: File) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      onImageSelected(file, dataUrl);
    };
    reader.readAsDataURL(file);
  }

  function openFileDialog() {
    fileInput.click();
  }
</script>

<div
  class="relative border-2 border-dashed rounded-xl p-8 text-center transition-colors duration-200 cursor-pointer
    {isDragging ? 'border-primary bg-primary/10' : 'border-slate-600 hover:border-slate-500'}"
  role="button"
  tabindex="0"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
  onclick={openFileDialog}
  onkeydown={(e) => e.key === 'Enter' && openFileDialog()}
>
  <input
    bind:this={fileInput}
    type="file"
    accept="image/*"
    class="hidden"
    onchange={handleFileSelect}
  />

  {#if previewUrl}
    <div class="flex flex-col items-center gap-4">
      <img
        src={previewUrl}
        alt="Preview"
        class="max-h-48 rounded-lg shadow-lg"
      />
      <p class="text-sm text-slate-400">Click or drag to change image</p>
    </div>
  {:else}
    <div class="flex flex-col items-center gap-4">
      <div class="w-16 h-16 rounded-full bg-background-lighter flex items-center justify-center">
        {#if isDragging}
          <ImageIcon class="w-8 h-8 text-primary" />
        {:else}
          <Upload class="w-8 h-8 text-slate-400" />
        {/if}
      </div>
      <div>
        <p class="text-white font-medium">Drop an image here</p>
        <p class="text-sm text-slate-400 mt-1">or click to browse</p>
      </div>
      <p class="text-xs text-slate-500">Supports JPG, PNG, WebP</p>
    </div>
  {/if}
</div>
