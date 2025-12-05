import argparse
import asyncio
from async_cat_api import AsyncCatImageProcessor

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["cat", "dog"], default="cat")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--output-dir", default="downloads")
    p.add_argument("--mode", choices=["standard", "pipeline"], default="standard", 
                   help="Use 'standard' for basic async/mp or 'pipeline' for generator pipeline")
    args = p.parse_args()
    
    proc = AsyncCatImageProcessor(provider=args.provider, output_dir=args.output_dir, limit=args.limit)
    
    if args.mode == "pipeline":
        print("Starting Async Generator Pipeline...")
        await proc.process_pipeline_generator()
    else:
        print("Starting Standard Async Processing...")
        await proc.process_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
