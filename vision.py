import cv2
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ---------------------------------------------------------
# 👁️ CONFIG
# ---------------------------------------------------------
# We use the 'base' model because it is fast (900MB) and accurate enough
MODEL_ID = "Salesforce/blip-image-captioning-base"
VIDEO_FOLDER = "project_data"

def get_video_path():
    """Finds the first MP4 file in the project_data folder."""
    if not os.path.exists(VIDEO_FOLDER):
        return None
    for file in os.listdir(VIDEO_FOLDER):
        if file.endswith(".mp4"):
            return os.path.join(VIDEO_FOLDER, file)
    return None

def extract_frame(video_path, seconds=10):
    """
    Extracts a single frame from the video at the specific timestamp.
    """
    print(f"🎞️ Extracting frame at {seconds} seconds...")
    cam = cv2.VideoCapture(video_path)
    
    # Get Frames Per Second (FPS) to calculate the exact frame number
    fps = cam.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * seconds)
    
    # Jump directly to that frame (Fast Seek)
    cam.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    ret, frame = cam.read()
    cam.release()
    
    if not ret:
        print("❌ Could not extract frame. Is the video too short?")
        return None
        
    # Convert from BGR (OpenCV standard) to RGB (AI standard)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def generate_caption(image):
    """
    Feeds the image into the BLIP transformer to get a text description.
    """
    print("🧠 Loading Vision Model (BLIP)...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    
    print("👁️ Analyzing image...")
    # Prepare the image
    inputs = processor(image, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    
    return description

if __name__ == "__main__":
    # 1. Find Video
    video_file = get_video_path()
    
    if video_file:
        print(f"📂 Found video: {video_file}")
        
        # 2. Extract Frame (Let's look at the 5-second mark)
        pil_image = extract_frame(video_file, seconds=5)
        
        if pil_image:
            # Optional: Save the frame to check it manually
            pil_image.save("test_frame.jpg")
            print("💾 Saved 'test_frame.jpg' for you to check.")
            
            # 3. Generate Description
            caption = generate_caption(pil_image)
            
            print("\n" + "="*40)
            print("👁️ VISUAL CONTEXT")
            print("="*40)
            print(f"The AI sees: '{caption}'")
            print("="*40)
    else:
        print("❌ No .mp4 file found in 'project_data'. Run main.py first.")