import os
import shutil
import requests

# ─── Configuration ─────────────────────────────────────────────────────────────
GITHUB_OWNER   = 'your-username'         # e.g. 'octocat'
GITHUB_REPO    = 'your-repo'             # e.g. 'my-images-repo'
GITHUB_BRANCH  = 'main'                  # e.g. 'main' or 'master'
GITHUB_PATH    = 'path/to/images'        # e.g. 'assets/pics'
LOCAL_DIR      = 'images'                # Local folder name
GITHUB_TOKEN   = None                    # Or 'ghp_…' to increase rate limits

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

def download_images():
    # Step 1: Remove existing folder if present
    if os.path.exists(LOCAL_DIR):
        print(f"Removing existing folder '{LOCAL_DIR}'…")
        shutil.rmtree(LOCAL_DIR)

    # Step 2: Recreate local folder
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Created folder '{LOCAL_DIR}'")

    # Step 3: Fetch directory listing from GitHub
    api_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/contents/{GITHUB_PATH}?ref={GITHUB_BRANCH}"
    )
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f"token {GITHUB_TOKEN}"

    resp = requests.get(api_url, headers=headers)
    resp.raise_for_status()
    items = resp.json()

    # Step 4: Download each image
    for entry in items:
        if entry.get('type') == 'file' and entry['name'].lower().endswith(IMAGE_EXTS):
            print(f"Downloading {entry['name']}…")
            img_data = requests.get(entry['download_url'], headers=headers).content
            with open(os.path.join(LOCAL_DIR, entry['name']), 'wb') as f:
                f.write(img_data)

    print("Done. All images saved in the 'images' folder.")


if __name__ == "__main__":
    download_images()