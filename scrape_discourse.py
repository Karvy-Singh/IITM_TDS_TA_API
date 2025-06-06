import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
import time
import json

# ─────── CONFIGURATION ───────
DISCOURSE_COOKIE = "EIeFAjTmAQY%2FFxjjaIU3XrlHVORE4nnDLN4JjSAMqi1F%2BudRE0MBeluvXdoWOMjFkbUj4Ousnr4t%2Fv7Ecs7QoFLRROriFMdjtBj7zlDh4KsxH0OKaoLzW4mN8TfX4sT%2Fs3l%2B3FNn%2BKk969BcrBFkeGm7LcjY5DOh%2F6BELLXdgcD3kM1vyIoWpOaTCfSxC2rJx4JrNu4PHyofOBacinWGLi%2BgzCUEk9857RKbvXuy2x5TuOsueCfp7034tqb9AAHbZ2TVAVUjR%2FqkEUz6BYBAXXtzkgK6mD1CmONV7MZO%2FVhLmDuz5nKgtPWqzqw%3D--DmNfpm4hhAJEZyKu--EZSqoSzTTPuSulaM8irWQQ%3D%3D"
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END   = datetime(2025, 4, 14, 23, 59, 59, tzinfo=timezone.utc)

session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})


def safe_get_json(url, retries=2):
    """GET a URL and return JSON, retrying up to `retries` times on failure."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"Warning: {url} returned status {r.status_code}.")
        except requests.RequestException as e:
            print(f"RequestException for {url}: {e}")
        time.sleep(2)
    return None


def get_topic_ids(category_slug="courses/tds-kb", category_id=34):
    """Return a list of topic dicts (each with id, title, created_at, etc.) until no more pages."""
    topics = []
    page = 0

    while True:
        url = f"{BASE_URL}/c/{category_slug}/{category_id}.json?page={page}"
        data = safe_get_json(url)
        if not data:
            break

        new_topics = data.get("topic_list", {}).get("topics", [])
        if not new_topics:
            break

        topics.extend(new_topics)
        print(f"Fetched {len(new_topics)} topics from page {page}.")
        page += 1
        time.sleep(0.1)

    return topics


def get_posts_in_topic(topic_id):
    """
    Fetch *all* posts in a given topic by paging through /t/{topic_id}/posts.json?page=N.
    Returns a list of dicts, each with topic metadata + post fields.
    """
    all_posts = []
    page = 0

    while True:
        url = f"{BASE_URL}/t/{topic_id}/posts.json?page={page}"
        data = safe_get_json(url)
        if not data:
            # Failed to fetch or no JSON back
            break

        posts = data.get("post_stream", {}).get("posts", [])
        if not posts:
            # No more posts on this page → we’re done
            break

        for post in posts:
            all_posts.append({
                "topic_id": topic_id,
                "topic_slug": data.get("slug", ""),
                "topic_title": data.get("title", ""),
                "topic_created_at": data.get("created_at", ""),
                "post_number": post["post_number"],
                "username": post["username"],
                "created_at": post["created_at"],
                "content": BeautifulSoup(post["cooked"], "html.parser").get_text(),
                "post_url": f"{BASE_URL}/t/{data.get('slug', '')}/{topic_id}/{post['post_number']}"
            })

        print(f"Fetched {len(posts)} posts from page {page} of topic {topic_id}.")
        page += 1
        time.sleep(1)

    return all_posts

import concurrent.futures

def fetch_posts_for_topic(topic):
    created_at = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
    if not (START <= created_at <= END):
        return []   # skip out‐of‐range topics
    return get_posts_in_topic(topic["id"])

def main():
    topics = get_topic_ids()
    print(f"Total topics: {len(topics)}")

    all_posts = []
    # Limit to, e.g., 10 threads at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit one future per topic
        futures = [executor.submit(fetch_posts_for_topic, topic) for topic in topics]

        for future in concurrent.futures.as_completed(futures):
            posts = future.result()
            if posts:
                all_posts.extend(posts)

    # Save as before
    with open("tds_discourse_posts.json", "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)
    print(f"Scraped {len(all_posts)} posts.")

if __name__ == "__main__":
    main()

