import os
import re
import json
import time
from datetime import datetime
from PIL import Image
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

def extract_all_text(root_dir: str, output_jsonl: str):
    # Initialize Gemini
    genai.configure(api_key="")
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite-preview-06-17")

    # Regex to pull out the post_id from filenames like "post123_1.png"
    pattern = re.compile(r"post(?P<post_id>\d+)_[^./]+\.(?:png|jpe?g|bmp|gif)$", re.IGNORECASE)

    # For rate‐limiting to 10 requests per minute
    request_count = 0
    start_time = time.time()

    print(f"[{datetime.now()}] Starting extraction: root={root_dir}, output={output_jsonl}")

    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        # Iterate each topic_id subdirectory
        for topic_id in os.listdir(root_dir):
            topic_path = os.path.join(root_dir, topic_id)
            if not os.path.isdir(topic_path):
                continue
            print(f"[{datetime.now()}] Entering topic: {topic_id}")

            # Iterate each image file in that topic
            for fname in os.listdir(topic_path):
                m = pattern.match(fname)
                if not m:
                    continue

                post_id = m.group("post_id")
                img_path = os.path.join(topic_path, fname)
                print(f"[{datetime.now()}] → Processing {img_path} (post_id={post_id})")

                # Rate-limit: max 10 calls per 60s
                request_count += 1
                elapsed = time.time() - start_time
                if request_count >= 10:
                    if elapsed < 60:
                        wait_sec = 60 - elapsed
                        print(f"[{datetime.now()}] Rate limit reached — sleeping {wait_sec:.1f}s")
                        time.sleep(wait_sec)
                    request_count = 0
                    start_time = time.time()

                img = Image.open(img_path)

                # API call with retry on ResourceExhausted
                try:
                    response = model.generate_content([img, "Extract all the text from this image."])
                except ResourceExhausted as e:
                    # if the exception carries a retry_delay, use it; else default to 60s
                    retry_delay = getattr(e, 'retry_delay', None)
                    wait = retry_delay.seconds if retry_delay and hasattr(retry_delay, 'seconds') else 70
                    print(f"[{datetime.now()}] Quota exceeded — retrying in {wait}s")
                    time.sleep(wait)
                    response = model.generate_content([img, "Extract all the text from this image."])

                try:
                    text = response.text.strip()
                except ValueError as e:
                    print(f"[{datetime.now()}] ⚠️ Skipping {img_path}: no text part in response (finish_reason={getattr(response, 'finish_reason', 'unknown')}).")
                    continue

                record = {
                    "topic_id": topic_id,
                    "post_id": post_id,
                    "data": text
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{datetime.now()}] ✔ Wrote record for post {post_id}")

    print(f"[{datetime.now()}] Done! Wrote results to {output_jsonl}")


if __name__ == "__main__":
    ROOT_DIR = "downloaded_images"
    OUTPUT_FILE = "extracted_text_3.jsonl"
    extract_all_text(ROOT_DIR, OUTPUT_FILE)

