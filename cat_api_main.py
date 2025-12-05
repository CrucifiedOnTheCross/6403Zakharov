import argparse

from cat_api import CatImageProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["cat", "dog"], default="cat")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--output-dir", default="downloads")
    args = p.parse_args()
    proc = CatImageProcessor(provider=args.provider, output_dir=args.output_dir, limit=args.limit)
    proc.process_and_save()


if __name__ == "__main__":
    main()
