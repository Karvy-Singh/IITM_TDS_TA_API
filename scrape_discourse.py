import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
import time
import json

# ─────── CONFIGURATION ───────
DISCOURSE_COOKIE = "20ozI4IGkalvVapK%2FXgviFGaMjRXRmC8xxeQUQBBw9i5JC1jjT%2FOvEQdi9XcMGmnIGlh6%2BfDoKgNxgA1A5ey%2FZwxsG3dKJ11KbLQXaT4fx6DhF%2FFbrXJpbhyDzzJIgDz%2F%2FfbKcSmBz%2BBg6xmng7jcimrU0aTyZVGfH2gfrOXe%2B%2F1tWCf%2F%2FjWZNOSyHVTGO%2BkucJncISb3I2s%2BHjz%2BKMRaV%2BrJOns0iCO1DTScxkCCxR8EetZflwY7ELf1vx%2FMZ9wKnDu4t2n9rWsUU7cDCnRm%2FkdBfNLpEE8e4vTyo9sbuQlGsQ1HNvNTZb9eE8%3D--7opu78I6aCsXaLDm--Zxnka70lrLCq7bC1hhfo8w%3D%3D"
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
        time.sleep(1)

    return topics


def get_posts_in_topic(topic_id):
    """Fetch all posts in a given topic, returning a list of post‐metadata dicts."""
    url = f"{BASE_URL}/t/{topic_id}.json"
    data = safe_get_json(url)
    if not data:
        return []

    topic_title = data.get("title", "")
    topic_slug  = data.get("slug", "")
    topic_created_at = data.get("created_at", "")

    out = []
    for post in data.get("post_stream", {}).get("posts", []):
        out.append({
            "topic_id": topic_id,
            "topic_slug": topic_slug,
            "topic_title": topic_title,
            "topic_created_at": topic_created_at,
            "post_number": post["post_number"],
            "username": post["username"],
            "created_at": post["created_at"],
            "content": BeautifulSoup(post["cooked"], "html.parser").get_text(),
            "post_url": f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post['post_number']}"
        })
    return out


def main():
    all_posts = []
    topics = get_topic_ids()

    print(f"Total topics collected: {len(topics)}")
    for topic in tqdm(topics):
        # Parse topic creation time
        created_at = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
        if not (START <= created_at <= END):
            continue

        posts = get_posts_in_topic(topic["id"])
        all_posts.extend(posts)

    # Save results
    with open("tds_discourse_posts.json", "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(all_posts)} posts.")


if __name__ == "__main__":
    main()

