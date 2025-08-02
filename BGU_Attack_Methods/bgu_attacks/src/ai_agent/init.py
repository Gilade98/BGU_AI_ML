import io
import os
from PIL import Image as PILImage

from src.ai_agent.graph import compiled_graph
import time
from src.config import TEST_JSON, LLM_OUTPUT_PATH, BATCH_SIZE, REASONINGS_PATH
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict

def merge_reasonings(existing_dict, new_dict):
    # Customize as needed; here new keys will override old ones
    merged = existing_dict.copy()
    merged.update(new_dict)
    return merged

def save_reasonings_batch(all_reasonings_dict):
    # Try to load existing reasonings if the file exists
    if os.path.exists(REASONINGS_PATH):
        with open(REASONINGS_PATH, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    else:
        existing = {}

    merged = merge_reasonings(existing, all_reasonings_dict)

    with open(REASONINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("Saved all reasonings to all_reasonings.json (merged)")

img = compiled_graph.get_graph().draw_mermaid_png()
PILImage.open(io.BytesIO(img)).show()

# ----- Load previous results -----
try:
    df = pd.read_csv(LLM_OUTPUT_PATH)
    processed_keys = set(str(i) for i in df["id"]) if not df.empty else set()
    print(f"Loaded {len(processed_keys)} previously processed entries.")
except FileNotFoundError:
    df = pd.DataFrame()
    processed_keys = set()
    print("No previous results found, starting fresh.")

# ----- Load test data -----
with open(TEST_JSON, "r", encoding="utf-8") as f:
    test_entries = json.load(f)

# ----- Main processing loop -----

results = []
n_processed = 0
n_skipped = 0
all_reasonings_dict = defaultdict(list)

for i, entry_dict in enumerate(tqdm(test_entries, desc="Processing")):
    entry_id = str(entry_dict.get("id"))
    if entry_id in processed_keys:
        print(f"Skipping ID {entry_id} (already processed)")   # <--- debug
        n_skipped += 1
        continue

    state = {
        "raw_cti_entry": entry_dict,
        "worker_outputs": [],
        "retrieval_worker_outputs": [],
    }
    start = time.time()
    result_state = compiled_graph.invoke(state)
    elapsed = time.time() - start
    print(f"Time : {elapsed:.2f} seconds")

    result_out = result_state["out"]
    row = {"id": entry_id, **result_out}
    results.append(row)
    n_processed += 1
    # --- Collect all reasonings
    all_reasonings = result_state["final_output"]["all_reasonings"]
    all_reasonings_dict[entry_id].extend(all_reasonings)

    # Save every BATCH_SIZE entries
    if n_processed % BATCH_SIZE == 0:
        print(f"Saving {len(results)} new results to CSV...")
        # Append to existing DataFrame and save
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        df.to_csv(LLM_OUTPUT_PATH, index=False)
        processed_keys.update(row["id"] for row in results)
        results = []

        save_reasonings_batch(all_reasonings_dict)

# Save any remaining
if results:
    print(f"Saving {len(results)} new results to CSV (final batch)...")
    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    df.to_csv(LLM_OUTPUT_PATH, index=False)
    processed_keys.update(row["id"] for row in results)

    # Save all reasonings dict after final batch
    save_reasonings_batch(all_reasonings_dict)

print(f"Done. Processed: {n_processed}, Skipped: {n_skipped}, Total: {len(df)}")