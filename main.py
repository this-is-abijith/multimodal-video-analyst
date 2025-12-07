import yt_dlp
import whisper
import torch
import os

# CONFIGURATION
OUTPUT_FOLDER = "project_data"
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

def download_audio_from_youtube(youtube_url):
    """
    Downloads the audio stream from a YouTube video and converts it to MP3.
    Returns: The file path of the downloaded mp3.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"⬇️ Starting download for: {youtube_url}")
    
    # yt-dlp configuration
    # UPDATED yt-dlp configuration for VIDEO
    ydl_opts = {
        'format': 'best[ext=mp4]',  # Download actual MP4 video
        'outtmpl': os.path.join(OUTPUT_FOLDER, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            # yt-dlp downloads as .webm/.m4a but converts to .mp3
            final_filename = os.path.splitext(filename)[0] + ".mp3"
            print(f"✅ Download Complete: {final_filename}")
            return final_filename
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None

def transcribe_with_whisper(audio_path):
    """
    Loads the Whisper model and transcribes the given audio file.
    """
    # Detect Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac M1/M2/M3 support
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"

    print(f"🧠 Loading Whisper model ('{MODEL_SIZE}') on device: {device.upper()}...")
    
    try:
        model = whisper.load_model(MODEL_SIZE, device=device)
        
        print("⏳ Transcribing... (This may take time depending on video length)")
        result = model.transcribe(audio_path)
        
        return result["text"]
    except Exception as e:
        print(f"❌ Transcription Failed: {e}")
        return None

if __name__ == "__main__":
    # Test with a short video to verify setup
    url = input("Enter a YouTube URL: ")
    
    # 1. Download
    audio_file = download_audio_from_youtube(url)
    
    # 2. Transcribe
    if audio_file:
        transcript = transcribe_with_whisper(audio_file)
        
        if transcript:
            print("\n--- TRANSCRIPT PREVIEW ---")
            print(transcript[:500] + "...") # Print first 500 chars
            
            # Save to file
            txt_path = audio_file.replace(".mp3", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"\n📄 Saved transcript to: {txt_path}")