import os
import json
import pandas as pd
from llama_cpp import Llama
import pyarrow.parquet as pq
import time


REAL_ESTATE_KEYWORDS = {
    "real estate", "housing market", "property", "mortgage", "rental", "home prices",
    "foreclosure", "realtor", "zoning", "construction", "landlord", "tenant",
    "apartment", "condo", "commercial real estate", "residential", "vacancy",
    "leasing", "buying", "selling", "investment property", "house", "loan",
    "interest rates", "homeowner", "valuation", "home sales", "urban planning",
    "housing policy", "affordable housing", "real estate development", "properties"
}

# Load Llama model (optimized for GPU)
llm = Llama.from_pretrained(
    repo_id="MaziyarPanahi/phi-4-GGUF",
    filename="phi-4.Q5_K_M.gguf",
    n_gpu_layers=-1,  # Full GPU acceleration
    n_ctx=2048,       # Max context length
    n_batch=1024,
    verbose=False
)


def get_last_processed_url(output_file):
    """Reads the last processed URL from the existing CSV file."""
    if not os.path.exists(output_file):
        return None  # Start from the beginning if no CSV exists
    # Use the actual column name "SOURCEURLS" from the parquet file
    df_out = pd.read_csv(output_file, usecols=["SOURCEURLS"])
    return df_out["SOURCEURLS"].iloc[-1] if not df_out.empty else None
    
def get_last_processed_row(output_file, input_file, batch_size=50000):
    if not os.path.exists(output_file):
        print(f"[DEBUG] Output file not found: {output_file}. Assuming no progress.", flush=True)
        return 0  
        
    df_out = pd.read_csv(output_file, usecols=["SOURCEURLS"])
    if df_out.empty:
        print(f"[DEBUG] Output file {output_file} is empty. Assuming no progress.", flush=True)
        return 0 

    last_processed_url = df_out["SOURCEURLS"].iloc[-1]  # Get last processed URL
    print(f"[DEBUG] Last processed URL: {last_processed_url}", flush=True)

    parquet_file = pq.ParquetFile(input_file)

    offset = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["SOURCEURLS"]):
        df_in  = batch.to_pandas()

        matches = df_in.index[df_in["SOURCEURLS"] == last_processed_url].tolist()

        if matches:
            row_in_batch = matches[-1] 
            global_idx = offset + row_in_batch
            return global_idx

        # If not found in this batch, move the offset forward.
        offset += len(df_in)

    return 0

def contains_real_estate_keywords(text):
    """Checks if the article contains any real estate-related keywords."""
    if text is None:
        return False 
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in REAL_ESTATE_KEYWORDS)

def process_article(row, llm):
    """Processes a single news article using Llama model and returns valid rows only."""
    #print(f"Processing article with SOURCEURLS: {row.get('SOURCEURLS', 'unknown')}", flush=True)
    if row["content"] is None or not contains_real_estate_keywords(row["content"]):
        return None  # Skip article if it doesn't contain real estate terms

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": """You are a helpful text-analysis assistant. Given the text below, please extract:
                1. Entities (people, organizations, places)
                2. Topics (key themes or categories)
                3. Sentiment (positive, negative, neutral)
                4. House prices (any mention of property costs, or "none" if no mention)
                5. Prediction (a short statement about the future or next steps implied by the text)

                Return your answer in valid JSON with the keys:
                "entities", "topics", "sentiment", "house_prices", "prediction"
                """},
                {"role": "user", "content": row["content"]}
            ],
            response_format={"type": "json_object"}
        )

        # Parse JSON response
        raw_output = response["choices"][0]["message"]["content"]
        #print(f"Raw response for SOURCEURLS {row.get('SOURCEURLS', 'unknown')}: {raw_output}", flush=True)
        result = json.loads(raw_output)
        return {
            "DATE": row["DATE"],
            "SOURCEURLS": row["SOURCEURLS"],
            "entities": ", ".join(result.get("entities", [])),
            "topics": ", ".join(result.get("topics", [])),
            "sentiment": result.get("sentiment", "unknown"),
            "house_prices": result.get("house_prices", "none"),
            "prediction": result.get("prediction", "No prediction")
        }
    except Exception:
        return None  # Exclude this row from output

def process_batch(batch_df, output_file, llm):
    """Processes a batch of articles and appends to CSV."""
    records = batch_df.to_dict("records")
    start_time = time.time()
    
    processed_data = []
    for row in records:
        result = process_article(row, llm)
        if result:
            processed_data.append(result)
        
    elapsed_time = time.time() - start_time
    print(f"Processed {len(processed_data)} rows in {elapsed_time:.2f} seconds.", flush=True)

    if not processed_data:
        print("No valid data in this batch, skipping CSV write.", flush=True)
        return

    # Convert results to DataFrame
    result_df = pd.DataFrame(processed_data)
    # Append results to CSV (or create if first batch)
    write_mode = "a" if os.path.exists(output_file) else "w"
    header = not os.path.exists(output_file)
    result_df.to_csv(output_file, mode=write_mode, header=header, index=False)

def analyze_llm_file(year, input_dir, output_dir, chunk_size=5000):
    """
    Processes real estate news articles using LLM and saves structured analysis.

    Args:
        year (int): Year of the dataset to process.
        input_dir (str): Directory containing input dataset.
        output_dir (str): Directory to save processed results.
        chunk_size (int): Number of articles processed per batch.
    """
    input_file = os.path.join(input_dir, f"gdelt_scraped_{year}.parquet")
    job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1")) - 1  # Convert to 0-indexed
    num_jobs = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    
    output_file = os.path.join(output_dir, f"news_analysis_{year}_job_{job_id+1}.csv")
    output_parquet = os.path.join(output_dir, f"news_analysis_{year}_job_{job_id+1}.parquet")


    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in the dataset: {total_rows}", flush=True)
    
    # Partition the rows among the jobs
    rows_per_job = total_rows // num_jobs
    start_idx = job_id * rows_per_job
    end_idx = total_rows if job_id == num_jobs - 1 else start_idx + rows_per_job
    print(f"Job {job_id+1}/{num_jobs} processing rows {start_idx} to {end_idx} out of {total_rows}.", flush=True)
    
    # --- Resuming Logic ---
    # Determine how many rows have already been processed in this partition.
    skip_rows_global = get_last_processed_row(output_file, input_file, batch_size=100_000)
    skip_rows = skip_rows_global - start_idx
    print(f"Resuming job {JOB_ID+1} from index {skip_rows_global}: already processed {skip_rows} rows.", flush=True)

    
    processed_count = 0  
    current_global_idx = 0
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        batch_size_actual = len(df)
        
        # Skip batches entirely before the partition start
        if current_global_idx + batch_size_actual <= start_idx:
            current_global_idx += batch_size_actual
            continue
        
        # If the batch starts before our partition, slice off the earlier rows
        if current_global_idx < start_idx:
            slice_start = start_idx - current_global_idx
            df = df.iloc[slice_start:]
            current_global_idx += slice_start
        
        # If this batch exceeds our partition, slice it to the partition end
        if current_global_idx + len(df) > end_idx:
            df = df.iloc[:end_idx - current_global_idx]
            current_global_idx = end_idx
        else:
            current_global_idx += len(df)
            
        # --- Resuming within Partition ---
        # If we still need to skip some rows (from a previous run), do so here.
        if skip_rows > 0:
            if skip_rows >= len(df):
                skip_rows -= len(df)
                processed_count += len(df)
                continue
            else:
                df = df.iloc[skip_rows:]
                processed_count += skip_rows
                skip_rows = 0

        
        print(f"Processing rows {processed_count + start_idx} to {processed_count + start_idx + len(df)}...", flush=True)
        process_batch(df, output_file, llm)
        processed_count += len(df)
        
        if current_global_idx >= end_idx:
            break

    print(f"Processing complete for {year}! Saving final CSV to Parquet...", flush=True)
    final_df = pd.read_csv(output_file)
    final_df.to_parquet(output_parquet, index=False)
    print(f"Final output saved as Parquet: {output_parquet}", flush=True)

if __name__ == "__main__":
    analyze_llm_file(
        year=2022,
        input_dir="",
        output_dir="",
        chunk_size=500
    )
