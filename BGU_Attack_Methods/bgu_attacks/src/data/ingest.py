import os
import json
import pandas as pd
from src.config import TACTIC_WHITELIST, OUT_PATH, RAW_PATH


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If keys are numeric strings ("0","1",...), convert dict to list
    if isinstance(data, dict) and list(data.keys())[0].isdigit():
        items = [(k, data[k]) for k in sorted(data, key=int)]
        return items
    else:
        return [("0", data)]


def flatten(entry, id):
    sigma = entry.get("sigma_rule", {}) or {}
    tags = sigma.get("tags", [])
    # Normalize tags: lower, hyphen→underscore
    norm_tags = [t.lower().replace("-", "_") for t in tags]
    # Tactics: strict whitelist, prefix stripped
    tactic_ids = [t.replace("attack.", "") for t in norm_tags if t in TACTIC_WHITELIST]
    # Techniques: startswith attack.t (original case)
    tech_ids = [t for t in tags if t.startswith("attack.t")]
    # Add more meta fields as needed
    return {
        "id": id,
        "url": entry.get("url", ""),
        "markdown": entry.get("markdown", ""),
        "sigma_tags": tags,
        "tech_ids": tech_ids,
        "tactic_ids": tactic_ids,
        "title": sigma.get('title', ''),
        "description": sigma.get('description', ''),
        "level": sigma.get('level', '').lower() if 'level' in sigma else '',
        "status": sigma.get('status', '').lower() if 'status' in sigma else '',
        "log_cat": sigma.get('logsource', {}).get('category', '') if isinstance(sigma.get('logsource'), dict) else '',
        "log_product": sigma.get('logsource', {}).get('product', '') if isinstance(sigma.get('logsource'), dict) else '',
        "has_known_fp": bool(sigma.get('falsepositives') and sigma['falsepositives'][0].lower() != 'unknown'),
    }

def build_dataframe(json_path):
    raw = load_json(json_path)
    records = [flatten(r, id=k) for k, r in raw]
    df = pd.DataFrame(records)
    # Clean: replace empty or whitespace-only markdown with pd.NA
    df['markdown'] = df['markdown'].replace(r'^\s*$', pd.NA, regex=True)
    return df

if __name__ == "__main__":
    # Make paths absolute from repo root
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = build_dataframe(RAW_PATH)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved cleaned parquet: {OUT_PATH}")
    print(df.head(2))
    print(f"Total rows: {len(df)}")

    # Optional: EDA stat - rows with more than 1 tactic label
    multi_label_count = (df['tactic_ids'].apply(len) > 1).sum()
    print(f"Rows with >1 tactic label: {multi_label_count} / {len(df)} ({100*multi_label_count/len(df):.1f}%)")
