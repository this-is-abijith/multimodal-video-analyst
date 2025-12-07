import streamlit as st
import yt_dlp
import whisper
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

# ---------------------------------------------------------
# ⚙️ CONFIGURATION
# ---------------------------------------------------------
DOWNLOAD_FOLDER = "app_downloads"
LLAMA_MODEL = "llama3.2"  # Using your local model

# Create folder if not exists
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# ---------------------------------------------------------
# 🛠️ CACHED MODELS (Loads once to save RAM)
# ---------------------------------------------------------

@st.cache_resource
def load_whisper():
    print("🧠 Loading Whisper...")
    return whisper.load_model("base")

@st.cache_resource
def load_vision_model():
    print("👁️ Loading BLIP Vision...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# ---------------------------------------------------------
# 🚀 CORE LOGIC
# ---------------------------------------------------------

def download_video(url):
    # Downloads MP4 video
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(DOWNLOAD_FOLDER, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename

def extract_key_frame(video_path):
    """Grabs a frame from the middle of the video."""
    cam = cv2.VideoCapture(video_path)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Jump to middle of video to get a good shot
    cam.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cam.read()
    cam.release()
    
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    return None

def generate_multimodal_summary(transcript, visual_desc):
    """
    Fuses Text + Vision into one prompt for Llama 3.
    """
    llm = ChatOllama(model=LLAMA_MODEL, temperature=0)
    
    # The "MSc Level" Prompt
    prompt = f"""
    You are an AI Video Analyst. Analyze this YouTube video based on two inputs:
    
    1. AUDIO TRANSCRIPT: "{transcript[:6000]}"... (truncated)
    2. VISUAL CONTEXT: The video shows "{visual_desc}"
    
    Task:
    - Summarize the video in 3 bullet points.
    - Mention if the visual context (what is seen) matches the audio topic (what is heard).
    - Give the video a creative title.
    """
    
    # We use 'invoke' (the new LangChain syntax)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ---------------------------------------------------------
# 🖥️ THE UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Video Analyst", page_icon="🧠")

st.title("🧠 Multimodel Video Analyst")
st.markdown("Returns a summary based on **what is heard** AND **what is seen**.")

# Input Box
url = st.text_input("Paste YouTube URL", "https://www.youtube.com/watch?v=jNQXAC9IVRw")

if st.button("Analyze Video"):
    # Status bar
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 1. Download
        status_text.text("⬇️ Step 1/4: Downloading video...")
        video_path = download_video(url)
        st.success(f"Downloaded: {os.path.basename(video_path)}")
        progress_bar.progress(25)
        
        # 2. Vision Analysis
        status_text.text("👁️ Step 2/4: Analyzing visual content...")
        processor, vision_model = load_vision_model()
        frame = extract_key_frame(video_path)
        
        if frame:
            st.image(frame, caption="Key Frame extracted by AI", width=300)
            
            # Generate Caption
            inputs = processor(frame, return_tensors="pt")
            out = vision_model.generate(**inputs)
            visual_text = processor.decode(out[0], skip_special_tokens=True)
            st.info(f"Visual Detection: **{visual_text}**")
        else:
            visual_text = "No visual context detected."
        progress_bar.progress(50)

        # 3. Audio Analysis
        status_text.text("👂 Step 3/4: Transcribing audio (Whisper)...")
        whisper_model = load_whisper()
        result = whisper_model.transcribe(video_path)
        transcript = result["text"]
        progress_bar.progress(75)
        
        # 4. Final Fusion
        status_text.text("🧠 Step 4/4: Fusing Multimodal Data (Llama 3)...")
        summary = generate_multimodal_summary(transcript, visual_text)
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")

        # OUTPUT
        st.divider()
        st.subheader("📝 Final Report")
        st.markdown(summary)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")