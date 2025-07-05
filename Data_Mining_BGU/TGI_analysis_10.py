#!/usr/bin/env python
import os
import json
import time
import socket
import subprocess
import asyncio
import pandas as pd
import pyarrow.parquet as pq
import requests
from huggingface_hub import AsyncInferenceClient

# Enable online Hugging Face look-ups
os.environ["HF_HUB_OFFLINE"] = "0"
print("HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"), flush=True)

# --- SLURM Job Partitioning ---
JOB_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", "1")) - 1  # Convert to 0-indexed
NUM_JOBS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

# --- Real Estate Keywords ---
REAL_ESTATE_KEYWORDS = {
    "real estate", "housing market", "property", "mortgage", "rental", "home prices",
    "foreclosure", "realtor", "zoning", "construction", "landlord", "tenant",
    "apartment", "condo", "commercial real estate", "residential", "vacancy",
    "leasing", "buying", "selling", "investment property", "house", "loan",
    "interest rates", "homeowner", "valuation", "home sales", "urban planning",
    "housing policy", "affordable housing", "real estate development", "properties"
}

# --- Concurrency Limit ---
CONCURRENT_LIMIT = 5

# --- Utility Functions for Port Handling ---
def is_port_free(port, host='localhost'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def get_free_port(start=3000, host='localhost'):
    """Ensure each SLURM job gets a unique port without conflicts."""
    base_port = start + JOB_ID  # Assign a unique base port per SLURM job
    port = base_port
    while not is_port_free(port, host):
        port += NUM_JOBS  # Increment in NUM_JOBS steps to reduce conflicts
    return port



def wait_for_server(url, timeout=300, interval=30):
    start_time = time.time()
    while True:
        try:
            # Send a minimal POST request with a dummy payload
            response = requests.post(url, json={"inputs": "ping"}, timeout=5)
            # Even if we get a 4xx error, we consider the server up if it doesn't error out
            if response.status_code < 500:
                print(f"Server is up at {url}.", flush=True)
                return True
        except Exception as e:
            print(f"Error in wait_for_server: {e}", flush=True)
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server did not start within {timeout} seconds")
        print(f"Waiting for server at {url} ...", flush=True)
        time.sleep(interval)


# --- Start the TGI Server ---
def start_tgi_server(model_id="microsoft/phi-4", quantize=True, base_port=3000):
    port = get_free_port(base_port)
    master_port = 29500 + JOB_ID  # Assigns a unique master port per SLURM job
    cmd = [
        "text-generation-launcher",
        "--model-id", model_id,
        "--port", str(port),
        "--hostname", "0.0.0.0",
        "--max-batch-size", "8",
        "--json-output",
        "--master-port", str(master_port),  # Explicitly assign master port
        "--disable-custom-kernels"
    ]
    if quantize:
        cmd.extend(["--quantize", "bitsandbytes"])
    print("Starting TGI server with command:", " ".join(cmd), flush=True)
    tgi_process = subprocess.Popen(cmd)
    server_url = f"http://localhost:{port}"
    # Wait until the /generate endpoint is available
    wait_for_server(server_url + "/v1/generate", timeout=900)
    return tgi_process, server_url

# --- Helper: Check if text contains real estate keywords ---
def contains_real_estate_keywords(text):
    if text is None:
        return False
    return any(keyword in text.lower() for keyword in REAL_ESTATE_KEYWORDS)

# --- Resume Functionality ---
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
    
def clean_markdown_json(text):
    """Remove markdown code fences (e.g. ```json ... ```) from the given text.
    If the response is truncated (no closing fence), just remove the opening fence."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = " ".join(lines).strip()
    return text

# --- Asynchronous Article Processing ---
async def process_article_async(row, client, max_new_tokens=300, semaphore=None):
    if row.get("content") is None or not contains_real_estate_keywords(row.get("content")):
        return None  # Skip irrelevant rows
    # Construct chat messages payload
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful text-analysis assistant. Given the text below, extract the following information and return your answer strictly as a JSON object with exactly these keys: "
                "\"entities\", \"topics\", \"sentiment\", \"house_prices\", \"prediction\". "
                "For 'entities', combine all entities mentioned in the text into a single flat list (do not separate into subcategories), and ensure that each element is a single-line string without newline characters. "
                "For 'topics', return a flat list of key topics or themes, with each list item on one line only. "
                "For 'sentiment', return a single string indicating overall sentiment (e.g., 'positive', 'negative', or 'neutral') on one line. "
                "For 'house_prices', return a string with property cost information or 'none' if not present, on one line. "
                "For 'prediction', return a brief prediction statement on one line. "
                "Do not include any extra commentary, markdown formatting, or code fences. "
                "Ensure that all double quotes within string values are properly escaped with a backslash. "
                "Ensure that all keys and values are correctly separated by commas, and that the JSON object is fully valid. "
                "Keep your entire output concise and under 1000 tokens. "
                "Your output must be a valid JSON object that looks exactly like this:\n"
                "{\"entities\": [\"entity1\", \"entity2\"], \"topics\": [\"topic1\", \"topic2\"], \"sentiment\": \"positive\", \"house_prices\": \"none\", \"prediction\": \"Your prediction here.\"}"
            )
        },
        {"role": "user", "content": row.get("content")}
    ]
    try:
        # Call the asynchronous chat completions API in streaming mode.
        if semaphore is not None:
            async with semaphore:
                stream = await client.chat.completions.create(
                    model="microsoft/phi-4",
                    messages=messages,
                    stream=True,
                    max_tokens=max_new_tokens,
                )
        else:
            stream = await client.chat.completions.create(
                model="microsoft/phi-4",
                messages=messages,
                stream=True,
                max_tokens=max_new_tokens,
            )
        output_text = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                output_text += chunk.choices[0].delta.content
        
        cleaned_output = clean_markdown_json(output_text)    
        # Debug
        #print("Raw output_text:", cleaned_output, flush=True)
                
        try:
            structured_data = json.loads(cleaned_output)
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}. Response was: {repr(cleaned_output)}", flush=True)
            raise
            
        return {
            "DATE": row.get("DATE"),
            "SOURCEURLS": row.get("SOURCEURLS"),
            "entities": ", ".join(structured_data.get("entities", [])),
            "topics": ", ".join(structured_data.get("topics", [])),
            "sentiment": structured_data.get("sentiment", "unknown"),
            "house_prices": structured_data.get("house_prices", "none"),
            "prediction": structured_data.get("prediction", "No prediction")
        }
    except Exception as e:
        print(f"Error processing article: {e}", flush=True)
        return None

async def process_batch_async(batch_df, output_file, client, semaphore, max_new_tokens=300):
    records = batch_df.to_dict("records")
    start_time = time.time()  # Start timing for this batch
    tasks = [asyncio.create_task(process_article_async(row, client, max_new_tokens, semaphore))
             for row in records]
    results = await asyncio.gather(*tasks)
    processed_data = [res for res in results if res is not None]
    if processed_data:
        result_df = pd.DataFrame(processed_data)
        num_rows_added = result_df.shape[0]
        result_df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
        print(f"Added {num_rows_added} rows to {output_file}", flush=True)
    else:
        print("No new rows processed.", flush=True)
    elapsed = time.time() - start_time  # End timing for this batch
    print(f"Batch processed in {elapsed:.2f} seconds.", flush=True)

# --- Main Analysis Function ---
async def analyze_llm_file(year, input_dir, output_dir, client, semaphore, chunk_size=500, max_new_tokens=300,
                           model_id="microsoft/phi-4", quantize=True):
    input_file = os.path.join(input_dir, f"gdelt_scraped_{year}.parquet")
    output_file = os.path.join(output_dir, f"news_analysis_{year}_job_{JOB_ID+1}.csv")
    output_parquet = os.path.join(output_dir, f"news_analysis_{year}_job_{JOB_ID+1}.parquet")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in dataset: {total_rows}", flush=True)
    
     # --- Divide Work Across Jobs ---
    rows_per_job = total_rows // NUM_JOBS
    start_idx = JOB_ID * rows_per_job
    end_idx = total_rows if JOB_ID == NUM_JOBS - 1 else (JOB_ID + 1) * rows_per_job
    print(f"Job {JOB_ID+1}/{NUM_JOBS} processing rows {start_idx} to {end_idx}.", flush=True)


    skip_rows_global = get_last_processed_row(output_file, input_file, batch_size=100_000)
    skip_rows = skip_rows_global - start_idx
    print(f"Resuming job {JOB_ID+1} from index {skip_rows_global}: already processed {skip_rows} rows.", flush=True)

    current_global_idx = 0  # Track global row index
    processed_count = 0     # Processed rows counter

    # Create the async inference client using the TGI server's /generate endpoint.
    # (The client is passed in from main_async.)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        batch_size_actual = len(df)
        if current_global_idx + batch_size_actual <= start_idx:
            current_global_idx += batch_size_actual
            continue
        if current_global_idx < start_idx:
            slice_start = start_idx - current_global_idx
            df = df.iloc[slice_start:]
            current_global_idx += slice_start
        if current_global_idx + len(df) > end_idx:
            df = df.iloc[:end_idx - current_global_idx]
            current_global_idx = end_idx
        else:
            current_global_idx += len(df)
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
        await process_batch_async(df, output_file, client, semaphore, max_new_tokens)
        processed_count += len(df)
        if current_global_idx >= end_idx:
            break

    print(f"Job {JOB_ID+1} complete! Saving final output...", flush=True)
    final_df = pd.read_csv(output_file)
    final_df.to_parquet(output_parquet, index=False)
    print(f"Final output saved as Parquet: {output_parquet}", flush=True)

# --- Main Async Wrapper ---
async def main_async():
    # Start the TGI server with the full model (microsoft/phi-4)
    tgi_process, server_url = start_tgi_server(model_id="microsoft/phi-4", quantize=False)
    print(f"TGI server started for job {JOB_ID+1} at {server_url}", flush=True)
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    try:
        # Create the async inference client pointing to the /generate endpoint
        client = AsyncInferenceClient(base_url=server_url +"/v1/")
        # Process your articles (adjust the directories and year as needed)
        await analyze_llm_file(
            2022,
            "",    # Input directory
            "",    # Output directory
            client,
            semaphore,
            chunk_size=500,
            max_new_tokens=2000,
            model_id="microsoft/phi-4",
            quantize=False
        )
    finally:
        print("Terminating TGI server...", flush=True)
        tgi_process.terminate()
        tgi_process.wait()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
  main()