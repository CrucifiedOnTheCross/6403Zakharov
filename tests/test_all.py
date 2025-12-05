import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cat_api import CatImageProcessor


def _san(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_")


def run():
    print("--- Part 1: Process and Save ---")
    proc1 = CatImageProcessor(provider="cat", output_dir="downloads", limit=2)
    proc1.process_and_save()

    print("--- Part 2: Sum and Diff ---")
    # Create a new processor with higher limit to ensure we get enough images
    proc2 = CatImageProcessor(provider="cat", output_dir="downloads", limit=5)
    images = proc2.fetch_images()
    print(f"Fetched {len(images)} images for sum/diff test")
    
    valid_images = [img for img in images if img.image is not None]
    
    if len(valid_images) >= 2:
        a, b = valid_images[0], valid_images[1]
        s = a + b
        d = a - b
        import os
        os.makedirs("downloads", exist_ok=True)
        path_s = os.path.join("downloads", f"sum_{_san(a.breed)}_{_san(b.breed)}.png")
        path_d = os.path.join("downloads", f"diff_{_san(a.breed)}_{_san(b.breed)}.png")
        s.save_original(path_s)
        d.save_original(path_d)
        print(f"Saved sum: {path_s}")
        print(f"Saved diff: {path_d}")
    else:
        print("Not enough images fetched for sum/diff test")


if __name__ == "__main__":
    run()
