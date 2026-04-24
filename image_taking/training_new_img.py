import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import random

def augment_medicine(
    input_images,       # list of 4 image paths
    medicine_name,      # e.g. "Pencor"
    output_dir,         # where to save
    target_count=100    # how many to generate
):
    """
    Generate target_count augmented images from input_images.
    Saves them ready for classifier training.
    """
    save_dir = os.path.join(output_dir, medicine_name)
    os.makedirs(save_dir, exist_ok=True)

    images = []
    for path in input_images:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    if not images:
        print(f"No images loaded for {medicine_name}")
        return

    count = 0
    while count < target_count:
        # Pick a random source image
        img = random.choice(images).copy()

        # ── Geometric transforms ──
        # Random rotation
        angle  = random.uniform(-180, 180)
        h, w   = img.shape[:2]
        M      = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img    = cv2.warpAffine(img, M, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)

        # Random flip
        flip = random.choice([-1, 0, 1, 2])  # 2 = no flip
        if flip != 2:
            img = cv2.flip(img, flip)

        # Random crop and resize back
        crop_pct = random.uniform(0.75, 1.0)
        ch       = int(h * crop_pct)
        cw       = int(w * crop_pct)
        y_start  = random.randint(0, h - ch)
        x_start  = random.randint(0, w - cw)
        img      = img[y_start:y_start+ch, x_start:x_start+cw]
        img      = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        # Random perspective warp — simulates different viewing angles
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        jitter = random.uniform(0, w * 0.1)
        pts2   = np.float32([
            [random.uniform(0, jitter),       random.uniform(0, jitter)],
            [random.uniform(w-jitter, w),     random.uniform(0, jitter)],
            [random.uniform(0, jitter),       random.uniform(h-jitter, h)],
            [random.uniform(w-jitter, w),     random.uniform(h-jitter, h)],
        ])
        M2  = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M2, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)

        # ── Color / brightness transforms ──
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Brightness
        brightness = ImageEnhance.Brightness(img_pil)
        img_pil    = brightness.enhance(random.uniform(0.5, 1.5))

        # Contrast
        contrast = ImageEnhance.Contrast(img_pil)
        img_pil  = contrast.enhance(random.uniform(0.7, 1.4))

        # Saturation
        saturation = ImageEnhance.Color(img_pil)
        img_pil    = saturation.enhance(random.uniform(0.7, 1.3))

        # Sharpness
        sharpness = ImageEnhance.Sharpness(img_pil)
        img_pil   = sharpness.enhance(random.uniform(0.5, 2.0))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # ── Noise ──
        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(5, 20),
                                     img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16) + noise,
                            0, 255).astype(np.uint8)

        # ── Blur ──
        if random.random() > 0.6:
            k   = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Save
        out_path = os.path.join(save_dir, f"{medicine_name}_{count:04d}.jpg")
        cv2.imwrite(out_path, img)
        count += 1

    print(f"  Generated {count} images for '{medicine_name}' → {save_dir}")


# ── Usage ──
augment_medicine(
    input_images=["sefloc1.jpg", "sefloc2.jpg",
                  "sefloc3.jpg", "sefloc4.jpg"],
    medicine_name="Pencor",
    output_dir="classifier_data/train/",
    target_count=50
)

# augment_medicine(
#     input_images=["prenolol_1.jpg", "prenolol_2.jpg",
#                   "prenolol_3.jpg", "prenolol_4.jpg"],
#     medicine_name="Prenolol_25",
#     output_dir="classifier_data/train/",
#     target_count=120
# )