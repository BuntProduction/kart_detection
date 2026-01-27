import argparse
import random
from pathlib import Path

from PIL import Image


def generate_random_crops(
    input_image: Path,
    output_dir: Path,
    count: int = 50,
    crop_size: int = 256,
    seed: int = 42,
    prefix: str = "no_kart_crop",
    quality: int = 95,
) -> None:
    if count <= 0:
        raise ValueError("count must be > 0")
    if crop_size <= 0:
        raise ValueError("crop_size must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_image).convert("RGB")

    # Ensure the image is large enough; upscale if needed.
    w, h = img.size
    if w < crop_size or h < crop_size:
        scale = max(crop_size / max(1, w), crop_size / max(1, h))
        new_w = max(crop_size, int(round(w * scale)))
        new_h = max(crop_size, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        w, h = img.size

    rng = random.Random(seed)

    max_x = w - crop_size
    max_y = h - crop_size

    for i in range(count):
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0
        crop = img.crop((x, y, x + crop_size, y + crop_size))

        out_path = output_dir / f"{prefix}_{i:03d}_x{x}_y{y}.jpg"
        crop.save(out_path, format="JPEG", quality=quality, optimize=True)

    print(f"Wrote {count} crops to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a small no_kart dataset by sampling random 256x256 crops from a single image."
    )
    parser.add_argument("--input", required=True, help="Path to the source image")
    parser.add_argument("--out", default="No kart/generated", help="Output directory")
    parser.add_argument("--count", type=int, default=50, help="Number of crops")
    parser.add_argument("--size", type=int, default=256, help="Crop size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prefix", default="no_kart_crop", help="Output filename prefix")
    args = parser.parse_args()

    generate_random_crops(
        input_image=Path(args.input),
        output_dir=Path(args.out),
        count=args.count,
        crop_size=args.size,
        seed=args.seed,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
