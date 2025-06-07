import requests
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ========== CONFIG ==========

COOKIES = {
    "_t": "xYGydsn3sg1RpkkzR919gL2x43%2BVGPWB%2BcaOlAC4HdFdFGR7F9nKBWweHVAscvlLdfnyyKEuvkFMWQnXLt6wkV8Sz3ouOFf%2B2S0XUgrR%2BDHjOcOgBcQXN7djsyfeZspIo8E6elzBk2aP3qzDOy7DO9TRN%2BsJsVBUBWK5avnONStdBXhFRfhX%2F%2FOmKemrY3rrRHruwsYPyG0Ji4ShdA%2BZ%2FczHmEa4W7LZ6NW8Q%2Bro6kwUzP0095b0PvOuv4sqd8gPqpE8PqP0Pnq5T5OsqQZ8UEWVaKLjnOcO6CdVmRaGVO9MMhVMK3VZhaYtLrs%3D--RKaBQC%2BLcKlj6P1z--BfS5jjv73HfKcJVDQtjJbA%3D%3D"
}

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

START_DATE = datetime(2025, 1, 1)
END_DATE   = datetime(2025, 4, 14)

OUTPUT_FILE = "discourse_filtered_posts.jsonl"

# How many threads to use for fetching post‐chunks
MAX_WORKERS = 4

# backoff_factor used by urllib3.Retry (for exponential backoff between retries)
BACKOFF_FACTOR = 1

# ========== SET UP SESSION WITH RETRY ==========

session = requests.Session()
session.headers.update(HEADERS)
session.cookies.update(COOKIES)

retry_strategy = Retry(
    total=5,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD", "OPTIONS"],
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def safe_get(url, **kwargs):
    """Wrapper around session.get to catch unexpected 429s and back off."""
    resp = session.get(url, **kwargs)
    if resp.status_code == 429:
        wait = int(resp.headers.get("Retry-After", 5))
        print(f"Received 429; backing off for {wait} seconds...")
        time.sleep(wait)
        resp = session.get(url, **kwargs)  # one more try
    resp.raise_for_status()
    return resp

# ========== HELPER FUNCTIONS ==========

def get_topic_urls(base_url):
    topic_urls = set()
    page = 0
    while True:
        url = f"{base_url}.json?page={page}"
        print(f"Fetching topic list page {page}...")
        data = safe_get(url).json()
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break
        for t in topics:
            topic_urls.add(f"https://discourse.onlinedegree.iitm.ac.in/t/{t['slug']}/{t['id']}")
        page += 1
        time.sleep(1)  # gentle paging
    return list(topic_urls)

def fetch_post_chunk(topic_id, slug, chunk):
    """Fetch up to 20 posts by ID in one request."""
    ids = "&".join(f"post_ids[]={pid}" for pid in chunk)
    url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/posts.json?{ids}"
    resp = safe_get(url)
    return resp.json().get("post_stream", {}).get("posts", [])

def parse_posts_from_topic(topic_url):
    slug, topic_id = topic_url.rstrip("/").rsplit("/", 1)
    meta = safe_get(f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json").json()
    post_ids = meta.get("post_stream", {}).get("stream", [])
    posts_data = []

    # break into 20‐ID chunks
    chunks = [post_ids[i:i+20] for i in range(0, len(post_ids), 20)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(fetch_post_chunk, topic_id, meta["slug"], chunk): chunk
            for chunk in chunks
        }
        for future in as_completed(future_to_chunk):
            for post in future.result():
                created = datetime.strptime(post["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                if START_DATE <= created <= END_DATE:
                    posts_data.append({
                        "id": post["id"],
                        "topic_id": post["topic_id"],
                        "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{meta['slug']}/{post['post_number']}",
                        "username": post.get("username"),
                        "content": BeautifulSoup(post["cooked"], "html.parser").get_text(),
                        "created_at": post["created_at"],
                    })
            time.sleep(0.2)  # small pause per‐chunk to be extra polite

    return posts_data

# ========== MAIN ==========

def main():
    all_posts = []
    topic_urls = get_topic_urls(BASE_URL)
    print(f"\nFound {len(topic_urls)} topics. Scraping posts...\n")

    for url in tqdm(topic_urls):
        try:
            all_posts.extend(parse_posts_from_topic(url))
        except Exception as e:
            print(f"Error on {url}: {e}")
        time.sleep(0.5)  # per‐topic delay

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_posts:
            json.dump(p, f)
            f.write("\n")

    print(f"\n✅ Done — {len(all_posts)} posts saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

