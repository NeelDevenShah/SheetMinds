#!/usr/bin/env python3
"""
Basic usage example for SheetMind - Agentic AI System for CSV/Excel Data Analysis
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import SheetMind from the installed package
from sheetmind import SheetMind

# Gemini API setup
GEMINI_API_KEY = "AIzaSyCOcVcE6yWZV4DntcEUqP52YBP7A-7S2nc"

class GeminiAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def analyze_with_gemini(self, prompt: str, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze data using Gemini API.
        
        Args:
            prompt: The analysis prompt
            data: Optional data to include in the analysis
            
        Returns:
            Analysis result as a string
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                api_key=self.api_key,
                temperature=0.3
            )
            
            # Format the data into the prompt if provided
            if data:
                data_str = json.dumps(data, indent=2)
                full_prompt = f"""Given the following data analysis:
                
                {data_str}
                
                {prompt}
                
                Please provide a clear and concise response."""
            else:
                full_prompt = prompt
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.invoke(full_prompt)
            )
            
            return response.content
            
        except Exception as e:
            return f"Error analyzing with Gemini: {str(e)}"

# Sample data for demonstration
SAMPLE_CSV = """name,age,city,salary
Alice,28,New York,85000
Bob,34,San Francisco,95000
Charlie,45,Seattle,110000
Diana,29,New York,90000
Ethan,38,Chicago,80000
Fiona,31,San Francisco,100000
George,27,Chicago,75000
Hannah,42,Seattle,115000
Ian,35,New York,92000
Jessica,31,Chicago,85000
"""

async def main():
    # Create a sample CSV file
    os.makedirs("data", exist_ok=True)
    sample_file = Path("data/employees.csv")
    sample_file.write_text(SAMPLE_CSV)
    
    print(f"📊 SheetMind - Data Analysis Demo")
    print("-" * 40)
    
    # Initialize Gemini Analyzer
    gemini = GeminiAnalyzer(api_key=GEMINI_API_KEY)
    
    try:
        # Initialize SheetMind
        print("🚀 Initializing SheetMind...")
        sheetmind = SheetMind()
        await sheetmind.initialize()
        
        # Example 1: Basic data profiling
        print("\n🔍 Running data profiling...")
        result = await sheetmind.process_file(str(sample_file), "profile data")
        
        if result["success"]:
            profile = result["result"]["profile"]
            print("\n📈 Data Profile:")
            print(f"- File: {profile['file_path']}")
            print(f"- Rows: {profile['num_rows']}")
            print(f"- Columns: {profile['num_columns']}")
            print(f"- Columns: {', '.join(profile['columns'])}")
            
            # Print memory usage
            mem_mb = profile['memory_usage']['total_mb']
            print(f"\n💾 Memory Usage: {mem_mb} MB")
            
            # Print numeric columns statistics
            if 'numeric_stats' in profile:
                print("\n🔢 Numeric Columns:")
                for col in profile['numeric_stats'].keys():
                    stats = profile['numeric_stats'][col]
                    print(f"\n  {col}:")
                    print(f"  - Type: {profile['dtypes'][col]}")
                    print(f"  - Missing: {profile['missing_values'][col]} ({profile['missing_percentage'][col]}%)")
                    print(f"  - Mean: {stats.get('mean', 'N/A'):.2f}")
                    print(f"  - Min: {stats.get('min', 'N/A')}")
                    print(f"  - 25%: {stats.get('25%', 'N/A')}")
                    print(f"  - 50%: {stats.get('50%', 'N/A')}")
                    print(f"  - 75%: {stats.get('75%', 'N/A')}")
                    print(f"  - Max: {stats.get('max', 'N/A')}")
            
            # Print categorical columns statistics
            if 'categorical_stats' in profile and profile['categorical_stats']:
                print("\n📊 Categorical Columns:")
                for col, stats in profile['categorical_stats'].items():
                    print(f"\n  {col}:")
                    print(f"  - Type: {profile['dtypes'][col]}")
                    print(f"  - Missing: {profile['missing_values'][col]} ({profile['missing_percentage'][col]}%)")
                    print(f"  - Unique values: {stats['unique_values']}")
                    print("  - Top values:")
                    for value, count in stats['top_values'].items():
                        freq = stats['freq'].get(value, 0)
                        print(f"    - {value}: {count} ({freq}%)")
                
                if 'top_values' in stats and stats['top_values']:
                    top_val = next(iter(stats['top_values'].items()))
                    print(f"  - Most common: {top_val[0]} (count: {top_val[1]})")
        else:
            print("\n❌ Failed to profile data:")
            print(result.get("error", "Unknown error"))
            if "traceback" in result:
                print("\nTraceback:")
                print(result["traceback"])
        
        # Example 2: Custom analysis with Gemini
        print("\n🤖 Running custom analysis with Gemini...")
        
        # Define the analysis code
        analysis_code = """
# Group by city and calculate average salary and count
result = df.groupby('city')['salary'].agg({
    'average_salary': 'mean',
    'employee_count': 'count'
}).round(2)

# Convert to a dictionary for the result
result = result.to_dict()
"""
        
        # Run the analysis
        analysis_result = await sheetmind.process_file(str(sample_file), analysis_code)
        
        if analysis_result["success"]:
            print("\n🏙️  Salary Analysis by City:")
            city_data = analysis_result["result"]
            
            # Format the data for display
            formatted_data = []
            for city in city_data['mean'].keys():
                formatted_data.append({
                    'City': city,
                    'Average Salary': f"${city_data['mean'][city]:,.2f}",
                    'Employee Count': city_data['count'][city]
                })
            
            # Print the table
            print("\n" + "-" * 50)
            print(f"{'City':<20} | {'Avg Salary':<15} | {'Employees'}")
            print("-" * 50)
            for item in formatted_data:
                print(f"{item['City']:<20} | {item['Average Salary']:<15} | {item['Employee Count']}")
            print("-" * 50)
            
            # Get insights from Gemini
            print("\n💡 Generating insights with Gemini...")
            prompt = """
            Based on the salary data by city, provide 2-3 key insights about the salary distribution 
            and any interesting patterns you notice. Focus on the differences between cities and 
            any potential correlations with the number of employees.
            """
            
            insights = await gemini.analyze_with_gemini(prompt, city_data)
            print("\n🔍 Insights:")
            print(insights)
            
        else:
            print("\n❌ Analysis error:")
            print(analysis_result.get("error", "Unknown error"))
            if "traceback" in analysis_result:
                print("\nTraceback:")
                print(analysis_result["traceback"])
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        if 'sheetmind' in locals():
            await sheetmind.shutdown()
        if sample_file.exists():
            sample_file.unlink()
        if 'analysis_file' in locals() and analysis_file.exists():
            analysis_file.unlink()
        
        print("✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())
