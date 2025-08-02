# Enrichment Project Dataset

*Experimental scraper*

## Introduction

Sigma rules are a community-driven, YAML-based format for defining SIEM detection logic in a vendor-agnostic way. They enable analysts to write a single detection and apply it across different platforms. This dataset is drawn from the official Sigma repository: [SigmaHQ/sigma](https://github.com/SigmaHQ/sigma).

In Windows environments, Sigma rules commonly reference MITRE ATT\&CK® techniques (e.g., `attack.t1059.001` for PowerShell). Typical CTI domains include [SocPrime](https://socprime.com) and [CrowdStrike](https://www.crowdstrike.com), which host high-quality threat intelligence write-ups.

## Sigma Rule Fields

Each Sigma rule is represented as a nested JSON object mirroring the original YAML fields:

* **id**: Unique identifier (e.g., `f3b1e9a2-...`).
* **title**: Human-readable rule name.
* **description**: Brief summary of the behavior being detected.
* **status**: Lifecycle state (e.g., `experimental`, `stable`).
* **logsource**: Data source definition, e.g.:

  ```yaml
  logsource:
    category: windows
    product: security
  ```
* **detection**: Search logic sections, e.g.:

  ```yaml
  detection:
    selection:
      EventID: 4688
      CommandLine|contains: 'powershell'
    condition: selection
  ```
* **tags**: List of MITRE ATT\&CK IDs (e.g., `attack.execution`, `attack.persistence`).

> **Note:** In the JSON files below, the `sigma_rule` field is a parsed JSON object (not a YAML string).

## JSON Structure

**Total records:** 765

Six files are provided under `project_enrichment/`:

1. **enrichment\_dataset.json** — full Markdown content (765 records)
2. **enrichment\_dataset\_short.json** — Markdown truncated to 100 words (765 records)
3. **enrichment\_dataset\_sample50.json** — random 50-record sample from the full dataset
4. **enrichment\_dataset\_short\_sample50.json** — same 50-record sample, with Markdown truncated to 100 words
5. **sample\_indices.json** — the integer indices (0–764) selected for the sample, plus the seed

All “dataset” files share this JSON mapping format:

```jsonc
{
  "0": { /* first entry */ },
  "1": { /* second entry */ },
  …
}
```

* **Top-level keys** are stringified indices (`"0"`, `"1"`, ...).
* **url**: original reference URL.
* **markdown**: full or truncated text excerpted from that URL.
* **sigma\_rule**: parsed JSON object matching the rule’s fields.
* **sigma\_rule\_category**: the file path/category under SigmaHQ for traceability.

**sample\_indices.json** has this shape:

```json
{
  "seed": 42,
  "indices": [228, 51, …, 310]
}
```

Used to locate the same records in `enrichment_dataset.json`.
