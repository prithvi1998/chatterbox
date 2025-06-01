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

    # Check input text file
    if not os.path.exists(input_text_path):
        print(f"Error: input text file '{input_text_path}' not found.")
        sys.exit(1)

    with open(input_text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        print(text)

    if not text:
        print("Error: input text file is empty.")
        sys.exit(1)

    # Check optional audio prompt
    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        print(f"Error: audio prompt file '{audio_prompt_path}' not found.")
        sys.exit(1)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    model = ChatterboxTTS.from_pretrained(device=device)

    import math
    MAX_CHARS = 450

    # Split the text into chunks of <= 450 characters
    chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]

    print(f"Total chunks to process: {len(chunks)}")

    for idx, chunk in enumerate(chunks, start=1):
        kwargs = {
            "text": chunk
        }
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = audio_prompt_path

        # Generate audio
        wav = model.generate(**kwargs)

        # Construct filename with suffix
        base, ext = os.path.splitext(output_audio_path)
        output_file = f"{base}/{base}_{idx}{ext}"

        # Save to file
        ta.save(output_file, wav, model.sr)
        print(f"Saved chunk {idx} to: {output_file}")

if __name__ == "__main__":
    main()