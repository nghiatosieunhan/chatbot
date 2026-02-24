import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('LANDING_AI_KEY') or os.getenv('VISION_AGENT_API_KEY')
if api_key:
    os.environ['VISION_AGENT_API_KEY'] = api_key
else:
    print("Error: API Key not found in .env file!")
    exit(1)

from agentic_doc.parse import parse

def master_clean(text):
    """
    Deep cleaning: Removes metadata, fixes line breaks, and strips Markdown.
    """
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\*\*|\*', '', text)    
    v_chars = "a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    text = re.sub(fr'(?<=[{v_chars},])\n+(?=[{v_chars}])', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def run_pipeline(input_folder="./data/raw", output_folder="./data/processed"):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    pdf_files = list(Path(input_folder).glob("*.pdf"))
    print(f"🚀 Start processing {len(pdf_files)} file PDF...")

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        try:
            result = parse(str(pdf))
            if not result: continue
            raw_markdown = result[0].markdown
            clean_text = master_clean(raw_markdown)
            txt_filename = output_path / f"{pdf.stem}.txt"
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(clean_text)
            
            print(f"Successfully saved: {txt_filename.name}")
            time.sleep(1) 
            
        except Exception as e:
            print(f"False to process {pdf.name}: {e}")

if __name__ == "__main__":
    run_pipeline()