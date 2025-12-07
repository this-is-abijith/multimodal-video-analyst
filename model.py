import sys
import os
print(f"Python is running from: {sys.executable}")
try:
    import langchain
    print(f"LangChain is installed at: {os.path.dirname(langchain.__file__)}")
except ImportError:
    print("❌ LangChain is definitely NOT installed in this environment.")
import os
from langchain_ollama import ChatOllama
from langchain.chains import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# ⚙️ CONFIG: Local Model Settings
# ---------------------------------------------------------
# "llama3" is the standard. Use "llama3.2" if your PC is slow.
MODEL_NAME = "llama3.2" 

def load_transcript(file_path):
    """
    Reads the raw text file created in Phase 1.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def generate_summary(text_content):
    """
    Uses your LOCAL Ollama model to summarize.
    """
    print(f"🧠 Initializing Local Model: {MODEL_NAME}...")
    print("   (Note: Local AI is slower than Cloud AI. Please be patient!)")
    
    # Initialize the Local Llama 3 Model
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # 1. Split text (Local models have smaller context windows, so we split safely)
    # We use smaller chunks (4000 chars) to be safe with local VRAM
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = text_splitter.create_documents([text_content])
    
    print(f"🧩 Split transcript into {len(docs)} chunks. Processing...")

    # 2. Define the Chain
    # 'map_reduce' is safer for local models to ensure they don't crash on long text
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    # 3. Run
    result = chain.invoke({"input_documents": docs})
    summary = result["output_text"]
    return summary

if __name__ == "__main__":
    # AUTOMATICALLY FIND THE TEXT FILE
    if not os.path.exists("project_data"):
        print("❌ Error: 'project_data' folder not found. Did you run Phase 1?")
    else:
        files = [f for f in os.listdir("project_data") if f.endswith(".txt")]
        if not files:
            print("❌ No transcript found! Run main.py (Phase 1) first.")
        else:
            # Pick the latest file
            latest_file = max([os.path.join("project_data", f) for f in files], key=os.path.getctime)
            print(f"📂 Found transcript: {latest_file}")
            
            # Load text
            raw_text = load_transcript(latest_file)
            
            print("⏳ Generating Summary locally... (Watch your terminal for progress)")
            try:
                final_summary = generate_summary(raw_text)
                
                print("\n" + "="*40)
                print("🌟 VIDEO SUMMARY (Generated Locally)")
                print("="*40)
                print(final_summary)
                print("="*40)
                
                # Save summary
                save_path = latest_file.replace(".txt", "_summary_local.txt")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(final_summary)
                print(f"💾 Summary saved to: {save_path}")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Tip: If the error is 'Connection refused', make sure the Ollama app is running in the background.")