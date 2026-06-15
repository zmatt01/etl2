from functions import run
import pandas as pd
import os

if __name__ == "__main__":
    # Run the ETL pipeline
    results = run(path="data/marketing/cc_marketing.csv")
    
    # Option 1: Print info about what was created (good for testing/debugging)
    print("✓ ETL pipeline completed successfully!\n")
    print("Generated DataFrames:")
    for table_name, df in results.items():
        print(f"  - {table_name:15s}: {df.shape[0]:6d} rows × {df.shape[1]:3d} columns")
    
    # Option 2: Save all tables to CSV files
    output_path = "data/marketing/output/"
    os.makedirs(output_path, exist_ok=True)
    
    for table_name, df in results.items():
        file_path = os.path.join(output_path, f"{table_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"\n✓ Saved {table_name} to {file_path}")