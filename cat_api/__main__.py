import argparse
import logging
import sys

from .processor import CatImageProcessor

logger = logging.getLogger("cat_app")


def main():
    try:
        logger.info("Program started (Sync execution)")
        p = argparse.ArgumentParser(prog="python -m cat_api")
        p.add_argument("--provider", choices=["cat", "dog"], default="cat")
        p.add_argument("--limit", type=int, default=1)
        p.add_argument("--output-dir", default="downloads")
        args = p.parse_args()
        proc = CatImageProcessor(provider=args.provider, output_dir=args.output_dir, limit=args.limit)
        proc.process_and_save()
        logger.info("Program finished successfully")
    except Exception as e:
        logger.error(f"Program failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
