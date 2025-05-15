#!/usr/bin/env python3
"""
Command Line Interface for SheetMind
"""
import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional

# Import directly from the module to avoid circular imports
from .sheetmind import SheetMind
from .version import __version__

def print_result(result: dict):
    """Print the result in a user-friendly format."""
    if result.get("success", False):
        print("✅ Operation completed successfully!")
        if "result" in result:
            if isinstance(result["result"], (dict, list)):
                print(json.dumps(result["result"], indent=2))
            else:
                print(result["result"])
    else:
        print("❌ Operation failed!")
        if "error" in result:
            print(f"Error: {result['error']}")
        if "traceback" in result:
            print("\nTraceback:")
            print(result["traceback"])

async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SheetMind - Agentic AI for Data Analysis")
    parser.add_argument(
        "file", 
        nargs="?",
        help="Input file to process (CSV/Excel)"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="profile data",
        help="Task to perform (default: profile data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save results (JSON format)",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"SheetMind {__version__}",
    )

    args = parser.parse_args()

    if not args.file:
        parser.print_help()
        return

    sheetmind = SheetMind()
    try:
        await sheetmind.initialize()
        
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

        print(f"🔍 Processing {file_path.name}...")
        result = await sheetmind.process_file(str(file_path), args.task)
        
        # Print the result
        print_result(result)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to {output_path}")
    
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
    
    finally:
        await sheetmind.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
