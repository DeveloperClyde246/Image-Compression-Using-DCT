import os
import cv2
import shutil

# Define which extensions to process
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

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

        new_dim = max(200, (idx - 1) * 24) #change dimension
        resized = cv2.resize(img, (new_dim, new_dim), interpolation=cv2.INTER_AREA)

        out_name = f"image{idx}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        print(f"{fname} → {out_name} ({new_dim} x {new_dim})")

if __name__ == "__main__":
    input_folder  = "input_folder"
    output_folder = "input_folder_2"
    resize_and_rename_to_jpg(input_folder, output_folder)
