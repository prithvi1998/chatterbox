import sys
import os
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

def main():
    # Usage check
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python generate_chatterbox.py <input.txt> <output.wav> [audio_prompt.wav]")
        sys.exit(1)

    input_text_path = "/kaggle/input/" + sys.argv[1]
    output_audio_path = sys.argv[2]
    audio_prompt_path = sys.argv[3] if len(sys.argv) == 4 else None

    if not os.path.exists(input_text_path):
        print(f"Error: input text file '{input_text_path}' not found.")
        sys.exit(1)

    with open(input_text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("Error: input text file is empty.")
        sys.exit(1)

    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        print(f"Error: audio prompt file '{audio_prompt_path}' not found.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = ChatterboxTTS.from_pretrained(device=device)

    MAX_CHARS = 450
    chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
    print(f"Total chunks to process: {len(chunks)}")

    base, ext = os.path.splitext(output_audio_path)
    temp_files = []
    all_audio = []

    for idx, chunk in enumerate(chunks, start=1):
        kwargs = {"text": chunk}
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = audio_prompt_path

        wav = model.generate(**kwargs)
        chunk_path = f"{base}_chunk_{idx}{ext}"
        ta.save(chunk_path, wav, model.sr)
        print(f"Saved chunk {idx} to: {chunk_path}")
        temp_files.append(chunk_path)

        # Load audio back into tensor for concatenation
        waveform, _ = ta.load(chunk_path)
        all_audio.append(waveform)

    # Concatenate all audio chunks
    final_audio = torch.cat(all_audio, dim=1)
    ta.save(output_audio_path, final_audio, model.sr)
    print(f"Final audio saved to: {output_audio_path}")

    # Clean up temporary chunk files
    for f in temp_files:
        os.remove(f)
    print("Temporary chunk files deleted.")

if __name__ == "__main__":
    main()
