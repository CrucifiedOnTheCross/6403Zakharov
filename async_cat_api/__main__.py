import argparse
import asyncio
import sys
import os
import logging

logger = logging.getLogger("cat_app")

from .async_processor import AsyncCatImageProcessor

async def main():
    try:
        logger.info("Program started (Package execution)")
        
        p = argparse.ArgumentParser(description="Async Cat Image Processor CLI")
        p.add_argument("--provider", choices=["cat", "dog"], default="cat", help="API Provider")
        p.add_argument("--limit", type=int, default=1, help="Number of images to process")
        p.add_argument("--output-dir", default="downloads", help="Directory to save images")
        p.add_argument("--mode", choices=["standard", "pipeline"], default="standard", 
                       help="Use 'standard' for basic async/mp or 'pipeline' for generator pipeline")
        args = p.parse_args()
        
        proc = AsyncCatImageProcessor(provider=args.provider, output_dir=args.output_dir, limit=args.limit)
        
        if args.mode == "pipeline":
            await proc.process_pipeline_generator()
        else:
            await proc.process_pipeline()
            
        logger.info("Program finished successfully")
        
    except Exception as e:
        logger.error(f"Program failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass