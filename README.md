# 🧠 Multimodal AI Video Analyst

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![AI](https://img.shields.io/badge/GenAI-Llama3%20%7C%20Whisper%20%7C%20BLIP-green)

An advanced AI application that "watches" YouTube videos to generate comprehensive summaries. Unlike standard summarizers that only read transcripts, this tool employs a **Multimodal RAG architecture** to analyze both **Audio** (Speech-to-Text) and **Visuals** (Computer Vision) simultaneously.

## 🚀 Key Features

* **Dual-Stream Analysis:** Processes audio and visual data in parallel streams.
* **Local LLM Inference:** Runs entirely offline using **Llama 3.2** (via Ollama) for privacy and zero cost.
* **Computer Vision:** Uses **Salesforce BLIP** (Bootstrapping Language-Image Pre-training) to caption keyframes and verify video context.
* **Speech Recognition:** Implements **OpenAI Whisper** for high-accuracy transcription.
* **Context Verification:** The AI detects mismatches between what is *said* versus what is *shown* (e.g., detecting clickbait thumbnails or unrelated visuals).

## 🏗️ Architecture

1.  **Ingestion:** `yt-dlp` downloads video data.
2.  **Audio Stream:** Audio is extracted and transcribed into text using **Whisper**.
3.  **Visual Stream:** **OpenCV** extracts keyframes; **BLIP** generates a textual description of the scene.
4.  **Fusion Layer:** **LangChain** combines the Transcript and Visual Description into a structured prompt.
5.  **Reasoning:** **Llama 3.2** generates the final summary, cross-referencing both data sources.

## 🛠️ Prerequisites

Before running the app, ensure you have the following installed on your system:

1.  **Python 3.10+**
2.  **FFmpeg:** Required for audio processing.
    * *Windows:* [Download here](https://www.gyan.dev/ffmpeg/builds/) and add to System PATH.
    * *Mac:* `brew install ffmpeg`
    * *Linux:* `sudo apt install ffmpeg`
3.  **Ollama:** Required to run the Llama 3 model locally.
    * Download from [ollama.com](https://ollama.com).

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/this-is-abijith/multimodal-video-analyst.git
    cd multimodal-video-analyst
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the LLM**
    Open your terminal and pull the Llama 3.2 model:
    ```bash
    ollama pull llama3.2
    ```

## ▶️ Usage

1.  Ensure the **Ollama app** is running in the background.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  The web interface will open in your browser (`http://localhost:8501`).
4.  Paste a YouTube URL and click **Analyze Video**.

## 📝 Example Output

**Input:** A video titled "Geography Challenge" showing a rubber duck on a map.

**AI Analysis Report:**
* **Visual Context:** "A person's hands are shown on a table with a yellow rubber duck."
* **Audio Context:** Discussion about a game called "Guess the Country."
* **Fusion Logic:** The AI successfully identified that the visual props (duck/map) were part of the game described in the audio.

## 📂 Project Structure

multimodal-video-analyst/ ├── app.py # Main application logic (Streamlit + AI Pipeline) ├── requirements.txt # Python dependencies ├── .gitignore # Ignored files (videos, models) └── README.md 

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License

This project is open-source and available under the MIT License.