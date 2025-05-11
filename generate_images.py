import os
import cv2
import shutil
import requests

# ─── Configuration ─────────────────────────────────────────────────────────────
GITHUB_OWNER   = 'DeveloperClyde246'         # e.g. 'octocat'
GITHUB_REPO    = 'Image-Compression-Using-DCT'             # e.g. 'my-images-repo'
GITHUB_BRANCH  = 'master'                  # e.g. 'main' or 'master'
GITHUB_PATH    = 'input_folder'        # e.g. 'assets/pics'
LOCAL_DIR      = 'images_from_github'                # Local folder name
GITHUB_TOKEN   = None                    # Or 'ghp_…' to increase rate limits

# Define which extensions to process
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

def download_images():
    # Step 1: Remove existing folder if present
    if os.path.exists(LOCAL_DIR):
        print(f"Removing existing folder '{LOCAL_DIR}'¡­")
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
            print(f"Downloading {entry['name']}­")
            img_data = requests.get(entry['download_url'], headers=headers).content
            with open(os.path.join(LOCAL_DIR, entry['name']), 'wb') as f:
                f.write(img_data)

    print("Done. All images saved in the 'images' folder.")



def resize_and_rename_to_jpg(input_dir, output_dir, max_images=50):
    # Clear output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Gather image files
    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    # Sort by file-size ascending
    files.sort(key=lambda f: os.path.getsize(os.path.join(input_dir, f)))

    # --- DEBUG: print sorted list with sizes ---
    print("Files sorted by size (smallest → largest):")
    for i, fname in enumerate(files, 1):
        size = os.path.getsize(os.path.join(input_dir, fname))
        print(f"{i:2d}. {fname} — {size} bytes")
    print("-" * 40)

    for idx, fname in enumerate(files[:max_images], start=1):
        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            print(f"⚠️ Skipping unreadable file: {fname}")
            continue

        new_dim = 200 + (idx - 1) * 21
        resized = cv2.resize(img, (new_dim, new_dim), interpolation=cv2.INTER_AREA)

        out_name = f"image{idx}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        print(f"{fname} → {out_name} ({new_dim} x {new_dim})")

if __name__ == "__main__":
    input_folder  = "images_from_github"
    output_folder = "resized_images"
    download_images()
    resize_and_rename_to_jpg(input_folder, output_folder)
