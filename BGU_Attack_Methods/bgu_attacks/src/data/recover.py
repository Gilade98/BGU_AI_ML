import pandas as pd
from collections import Counter
import json
from src.config import DATA_DIR, REPO_ROOT
import os

def norm(s):
    # Converts pd.NA/None/float nan to "", then normalizes
    if pd.isna(s):
        return ""
    return str(s).strip().replace("\n", " ").replace("\r", "").lower()

def make_flat_key(row):
    return (
        norm(row["url"]),
        norm(row["title"]),
        norm(row["description"]),
        norm(row["markdown"]),
    )

def make_json_key(entry):
    sigma = entry.get("sigma_rule", {}) or {}
    return (
        norm(entry.get("url", "")),
        norm(sigma.get("title", "")),
        norm(sigma.get("description", "")),
        norm(entry.get("markdown", "")),
    )


if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    for split in splits:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{split}.parquet"))
        ids = df["id"].tolist()  # Get the id column as a list, in the same order as keys
        keys = df.apply(make_flat_key, axis=1).tolist()

        # keys is a list of composite keys from your df
        dupes = [item for item, count in Counter(keys).items() if count > 1]
        print(f"Duplicate keys found: {len(dupes)}")
        if dupes:
            print("Example dupe key(s):", dupes[:3])

        enrichment_path = os.path.join(REPO_ROOT,"data/raw/enrichment_dataset.json")
        with open(enrichment_path, "r", encoding="utf-8") as f:
            cti_entries = json.load(f)
            if isinstance(cti_entries, dict) and list(cti_entries.keys())[0].isdigit():
                cti_entries = [cti_entries[k] for k in sorted(cti_entries, key=int)]

        json_keys = [make_json_key(e) for e in cti_entries]

        # Build a map from composite key to full nested JSON entry
        json_key_to_entry = {k: e for k, e in zip(json_keys, cti_entries)}

        json_dupes = [item for item, count in Counter(json_keys).items() if count > 1]
        print(f"JSON duplicate keys found: {len(json_dupes)}")
        if json_dupes:
            print("Example JSON dupe key(s):", json_dupes[:3])

        # Now select all entries using their keys
        entries = []
        for _id, k in zip(ids, keys):
            if k in json_key_to_entry:
                entry = dict(json_key_to_entry[k])  # make a shallow copy
                entry["id"] = _id
                entries.append(entry)

        print(f"Recovered {len(entries)} entries from original JSON.")

        print(f"df rows: {len(df)}")
        print(f"Recovered entries: {len(entries)}")

        # save
        out_path = f"{REPO_ROOT}/data/raw/splits/{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Saved to {out_path}")
