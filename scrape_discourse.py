import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from tqdm import tqdm

DISCOURSE_COOKIE = "YOUR_DISCOURSE_SESSION_COOKIE_HERE"  # <--- PUT YOUR COOKIE HERE
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 4, 14, 23, 59, 59, tzinfo=timezone.utc)

HEADERS = {"User-Agent": "Mozilla/5.0"}
CONCURRENT_REQUESTS = 10  # Reduce if you hit rate limits often

async def fetch_json(session, url, max_retries=5, **kwargs):
    retry_count = 0
    while retry_count < max_retries:
        async with session.get(url, **kwargs) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 60
                print(f"Received 429 for {url}. Waiting {wait_time} seconds before retrying... (retry {retry_count+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                retry_count += 1
            else:
                print(f"Failed GET {url}: {resp.status}")
                return None
    print(f"Max retries exceeded for {url}")
    return None

async def fetch_all_topics(session):
    print("Fetching topics...")
    topics = []
    page = 0
    while True:
        url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}.json?page={page}"
        data = await fetch_json(session, url)
        if not data:
            break
        topic_list = data.get("topic_list", {}).get("topics", [])
        if not topic_list:
            break
        topics.extend(topic_list)
        print(f"Fetched {len(topic_list)} topics from page {page}")
        page += 1
    print(f"Total topics fetched: {len(topics)}")
    return topics

async def fetch_topic_posts(session, topic_id):
    url = f"{BASE_URL}/t/{topic_id}.json"
    topic_json = await fetch_json(session, url)
    if not topic_json:
        return []
    posts_count = topic_json['posts_count']
    post_ids = list(range(1, posts_count + 1))

    posts = []
    BATCH_SIZE = 50  # Discourse default per request
    for i in range(0, len(post_ids), BATCH_SIZE):
        nums = post_ids[i:i+BATCH_SIZE]
        nums_param = ",".join(str(num) for num in nums)
        posts_url = f"{BASE_URL}/t/{topic_id}/posts.json?post_numbers={nums_param}"
        posts_json = await fetch_json(session, posts_url)
        if not posts_json:
            continue
        these_posts = posts_json.get("post_stream", {}).get("posts", [])
        posts.extend(these_posts)
    return posts

def filter_posts_by_date(posts):
    filtered = []
    for post in posts:
        created = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
        if START_DATE <= created <= END_DATE:
            filtered.append(post)
    return filtered

async def main():
    cookies = {"_t": DISCOURSE_COOKIE}
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(headers=HEADERS, cookies=cookies, connector=connector) as session:
        topics = await fetch_all_topics(session)
        all_posts = []

        sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def fetch_and_process_topic(topic):
            async with sem:
                topic_id = topic["id"]
                topic_title = topic["title"]
                posts = await fetch_topic_posts(session, topic_id)
                posts_in_range = filter_posts_by_date(posts)
                if posts_in_range:
                    print(f"Topic '{topic_title}': {len(posts_in_range)} posts in date range.")
                    for post in posts_in_range:
                        all_posts.append({
                            "topic_id": topic_id,
                            "topic_title": topic_title,
                            "post_number": post["post_number"],
                            "username": post["username"],
                            "created_at": post["created_at"],
                            "cooked": post["cooked"],
                            "raw": post.get("raw", ""),
                        })

        tasks = [fetch_and_process_topic(topic) for topic in topics]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing topics"):
            await f

        print(f"\nSaving {len(all_posts)} posts to 'discourse_posts.json'...")
        with open("discourse_posts.json", "w", encoding="utf-8") as f:
            json.dump(all_posts, f, indent=2, ensure_ascii=False)
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
