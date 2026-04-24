from ultralytics import YOLO
import os
import shutil

def auto_label_generated_images(
    image_folder=r"C:\Users\pasul\Desktop\IternStuff\Medicine_detection_Webapp\Sefloc",    # folder with your 50 generated Sefloc images
    model_path="best.pt",
    conf=0.4
):
    """
    Auto-label generated images using existing YOLO model.
    Saves images + label files ready to upload to Roboflow.
    """
    model      = YOLO(model_path)
    labels_dir = os.path.join(image_folder, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    img_files = [f for f in os.listdir(image_folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"  Auto-labeling {len(img_files)} images...")

    success = 0
    failed  = []

    for img_file in img_files:
        img_path = os.path.join(image_folder, img_file)
        results  = model(img_path, conf=conf, task='obb', verbose=False)

        label_path = os.path.join(
            labels_dir,
            os.path.splitext(img_file)[0] + ".txt"
        )

        boxes_found = 0
        with open(label_path, 'w') as f:
            for r in results:
                if r.obb is None:
                    continue
                for box in r.obb:
                    cls  = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    # Normalized OBB points for YOLO format
                    pts  = box.xyxyxyxyn[0].cpu().numpy().flatten()
                    line = f"{cls} " + " ".join(f"{p:.6f}" for p in pts)
                    f.write(line + "\n")
                    boxes_found += 1

        if boxes_found > 0:
            success += 1
            print(f"  ✓ {img_file} → {boxes_found} box(es) found")
        else:
            failed.append(img_file)
            print(f"  ✗ {img_file} → nothing detected")

    print(f"\n  Done: {success}/{len(img_files)} labeled")
    if failed:
        print(f"  Failed ({len(failed)} images — label manually):")
        for f in failed:
            print(f"    {f}")

    print(f"\n  Upload to Roboflow:")
    print(f"    Images: {image_folder}")
    print(f"    Labels: {labels_dir}")

# Usage
auto_label_generated_images(
    image_folder="classifier_data/train/Sefloc/",
    model_path="best.pt",
    conf=0.4
)