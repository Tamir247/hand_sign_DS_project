import os
import shutil
from pathlib import Path

# RDNA2 (RX 6700 XT / gfx1031) requires this override for ROCm wheels.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

from ultralytics import YOLO


def ensure_val_split(data_root: Path) -> None:
    """YOLOv8 expects a 'val' folder; create a symlink if only 'valid' exists."""
    val_dir = data_root / "val"
    valid_dir = data_root / "valid"
    if not val_dir.exists() and valid_dir.exists():
        val_dir.symlink_to(valid_dir.resolve())
        print(f"Created symlink: {val_dir} -> {valid_dir}")


def copy_best_weights(run_dir: Path, output_root: Path) -> None:
    weights_dir = run_dir / "weights"
    for name in ("best.pt", "last.pt"):
        src = weights_dir / name
        if src.exists():
            shutil.copy2(src, output_root / name)
            print(f"Copied {name} to: {output_root / name}")


def train():
    # ---- Edit these values directly to change the run ----
    data_root   = Path("hand-sign-drive").resolve()
    output_dir  = Path("outputs_yolo_classification").resolve()
    model_name  = "yolov8n-cls.pt"
    epochs      = 50
    batch_size  = 32
    lr          = 1e-3
    img_size    = 224
    num_workers = 4
    device      = "0"      # "0" = ROCm GPU, "cpu" = force CPU
    seed        = 42
    # ------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_val_split(data_root)

    model = YOLO(model_name)

    train_kwargs = dict(
        data=str(data_root),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr,
        workers=num_workers,
        device=device,
        seed=seed,
        project=str(output_dir),
        name="yolo_cls",
        exist_ok=True,
        verbose=False,
        patience=20,
        cos_lr=True,
        weight_decay=0.0005,
        # Augmentation
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.2,
        auto_augment="randaugment",
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as exc:
        if "invalid device function" in str(exc).lower():
            print("ROCm kernel launch failed. Retrying on CPU...")
            train_kwargs["device"] = "cpu"
            results = model.train(**train_kwargs)
        else:
            raise

    run_dir = Path(results.save_dir)
    copy_best_weights(run_dir=run_dir, output_root=output_dir)
    print(f"\nTraining complete. Run directory: {run_dir}")

    # Evaluate on test set
    best_pt = output_dir / "best.pt"
    if best_pt.exists() and (data_root / "test").exists():
        print("\nEvaluating on test set...")
        best_model = YOLO(str(best_pt))
        try:
            test_results = best_model.val(data=str(data_root), split="test", device=device)
        except RuntimeError as exc:
            if "invalid device function" in str(exc).lower():
                print("ROCm kernel launch failed during val. Retrying on CPU...")
                test_results = best_model.val(data=str(data_root), split="test", device="cpu")
            else:
                raise
        print(f"Test results: {test_results}")


if __name__ == "__main__":
    train()
