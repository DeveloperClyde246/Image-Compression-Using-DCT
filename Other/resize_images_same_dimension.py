import os
import cv2
import shutil

# Extensions to include
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

def resize_sort_and_rename(input_dir, output_dir, target_size):
    """
    1. Resize every image in input_dir to target_size (width, height).
    2. Save resized copies in output_dir.
    3. Sort those by file-size ascending.
    4. Rename them to image1.ext, image2.ext, ... in output_dir.
    """

    # Clear output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 2) Gather and resize
    temp_files = []
    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue

        src = os.path.join(input_dir, fname)
        img = cv2.imread(src)
        if img is None:
            print(f"⚠️ Could not read {fname}, skipping.")
            continue

        # Resize to uniform dimensions
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Save resized image under its original name
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, resized)
        temp_files.append(fname)

    if not temp_files:
        print("No images found to process.")
        return

    # 3) Sort by file-size (smallest → largest)
    temp_files.sort(key=lambda f: 
        os.path.getsize(os.path.join(output_dir, f))
    )

    # 4) Rename in sequence
    for idx, old_name in enumerate(temp_files, start=1):
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"image{idx}{ext}"
        src = os.path.join(output_dir, old_name)
        dst = os.path.join(output_dir, new_name)
        os.rename(src, dst)
        print(f"✅ {old_name} → {new_name}")

if __name__ == "__main__":
    # ─── USER CONFIG ─────────────────────────────────────
    input_folder  = "input_folder"
    output_folder = "same_dimension_output_folder"
    # e.g., all images become 256×256 pixels:
    target_size   = (1200, 1200)  
    # ────────────────────────────────────────────────────

    resize_sort_and_rename(input_folder, output_folder, target_size)
