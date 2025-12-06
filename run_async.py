import argparse
import asyncio
from async_cat_api import AsyncCatImageProcessor

async def main():
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

if __name__ == "__main__":
    asyncio.run(main())
