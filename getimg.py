#!/usr/bin/env python3
import os
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
COOKIES = {
    # your login cookie here, if needed
    "_t": "eVbO%2FPgtUX3L7a%2BOIHjV3AaB1Cusxrb5vZc%2FX6RgsySgMsPEpxJMghc1k8xdKcn6OC4j1rZTb%2BsJKXik1pV4eR%2Ff%2BknUB2LWcz494IBEaTTBYGaLAkxxa%2BcZD%2B3hjmYg1lyPzrAaeX%2FMdi8qTka%2FkDS9y7PNHExDxPR8MUFpJ1WjYnDe7v1co3VJ25WaCymTBUanNSntCUHDViUFr0HbGT3KrP7NF1ikQSUuE10JIkDC%2BMAwZYYwoeb05LiJij2WEFofONfLtZu1aCwTcYbeVXp1SgDBWaewzLByMc4khIm%2F67vT9w3FwTiGGhA%3D--3htdf8IB0EyQ6HEk--aVoYi5S%2Fv2pPu3f9D4qxaQ%3D%3D"
}
HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

session = requests.Session()
session.headers.update(HEADERS)
session.cookies.update(COOKIES)

def get_topic_meta(topic_id):
    """Fetch first 20 posts + full post-ID stream."""
    url = f"{BASE_URL}/t/{topic_id}.json"
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json()

def get_posts_by_ids(topic_id, post_ids):
    """Fetch up to ~300 posts at a time by their internal IDs."""
    url = f"{BASE_URL}/t/{topic_id}/posts.json"
    params = [("post_ids[]", pid) for pid in post_ids]
    resp = session.get(url, params=params)
    resp.raise_for_status()
    return resp.json()["post_stream"]["posts"]

def collect_images_from_posts(posts):
    """
    Given a list of post dicts (each with 'post_number' and 'cooked'),
    return a list of {post_number, image_url, alt_text}.
    """
    images = []
    for post in posts:
        post_num = post.get("post_number")
        soup = BeautifulSoup(post.get("cooked", ""), "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src")
            alt = (img.get("alt") or "").strip()
            if not src:
                continue
            # exclude emojis
            if "emoji.discourse-cdn.com" in src or (alt.startswith(":") and alt.endswith(":")):
                continue
            # only "image" or "Screenshot…"
            if alt.lower() == "image" or alt.startswith("Screenshot"):
                images.append({
                    "post_number": post_num,
                    "image_url": src,
                    "alt_text": alt
                })
    return images

def download_images(images, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    for idx, img in enumerate(images, start=1):
        resp = session.get(img["image_url"], stream=True)
        resp.raise_for_status()
        ext = os.path.splitext(img["image_url"])[1].split("?")[0] or ".jpg"
        fname = f"post{img['post_number']}_{idx}{ext}"
        path = os.path.join(download_dir, fname)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded post #{img['post_number']} → {fname}")

def process_topic(topic_id):
    print(f"\n▶ Topic {topic_id}")
    meta = get_topic_meta(topic_id)
    initial_posts = meta["post_stream"]["posts"]
    all_ids = meta["post_stream"]["stream"]

    # Collect images from the first 20
    images = collect_images_from_posts(initial_posts)

    # Figure out which IDs remain … the stream is in ID order
    remaining_ids = all_ids[len(initial_posts):]

    # Fetch the rest in batches of 300
    for i in range(0, len(remaining_ids), 300):
        batch = remaining_ids[i : i + 300]
        print(f"  • fetching posts {i+1}–{i+len(batch)} of {len(all_ids)}…")
        more_posts = get_posts_by_ids(topic_id, batch)
        images.extend(collect_images_from_posts(more_posts))
        time.sleep(1)  # respect global rate limits

    if images:
        print(f"  → Found {len(images)} images; downloading…")
        download_images(images, download_dir=str(topic_id))
    else:
        print("  → No matching images found.")

def main():
    topics = [
        171054, 167471, 163381, 165433, 168506, 168515, 169029, 165959, 168011,
        168017, 169045, 168537, 166498, 141413, 164460, 164462, 168567, 168057,
        162425, 171141, 170131, 171668, 171672, 167072, 170147, 166576, 166593,
        168142, 168143, 166100, 172246, 165593, 172254, 166634, 166647, 164089,
        166651, 167679, 167172, 167699, 171798, 165142, 169247, 23335, 166189,
        172333, 161071, 161072, 164147, 165687, 161083, 169283, 170309, 163144,
        163147, 169807, 166738, 172373, 163158, 161120, 164205, 168303, 165746,
        164214, 168310, 168825, 168832, 164737, 169352, 163224, 169369, 171422,
        166303, 166816, 169888, 171428, 163241, 170413, 163247, 167344, 169393,
        163765, 164277, 172471, 161214, 168384, 164291, 168901, 167878, 165830,
        166349, 171473, 166866, 172497, 168916, 171477, 166357, 171485, 171999,
        166891, 171500, 168943, 169456, 167410, 172021, 167415, 171515, 99838,
        165396
    ]
    for tid in topics:
        process_topic(tid)
        # no extra sleep here; batch-level sleep is sufficient

if __name__ == "__main__":
    main()

