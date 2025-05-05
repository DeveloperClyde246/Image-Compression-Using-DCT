import os
import sys

# 1) Configuration: change this to your folder
INPUT_DIR = r"C:\Users\KEYU\Documents\GitHub\Image-Compression-Using-DCT\input_folder_unrenamed"

# 2) Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

def sort_and_rename_by_size(folder):
    # List all image files
    files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    if not files:
        print("No images found in:", folder)
        return

    # Sort by file size (ascending)
    files.sort(key=lambda fn: os.path.getsize(os.path.join(folder, fn)))

    # Rename in-place
    for idx, fname in enumerate(files, start=1):
        base, ext = os.path.splitext(fname)
        new_name = f"image{idx}{ext.lower()}"
        src = os.path.join(folder, fname)
        dst = os.path.join(folder, new_name)

        # Handle potential name conflicts
        if os.path.exists(dst):
            print(f"⚠️  Destination already exists, skipping: {dst}")
            continue

        print(f"Renaming '{fname}' → '{new_name}'")
        os.rename(src, dst)

if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print("ERROR: Input directory does not exist:", INPUT_DIR)
        sys.exit(1)
    sort_and_rename_by_size(INPUT_DIR)
    print("Done!")
