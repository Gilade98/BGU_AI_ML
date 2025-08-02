from langchain.tools import tool
from src.config import *

@tool("fetch_article", return_direct=True)
def fetch_article(url: str) -> str:
    """Fetches the content at the given URL as text (HTML)."""
    import requests
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.text[:100_000]
    except Exception as e:
        return f"ERROR: {str(e)}"


def embed_text_secbert(
    text,
    tokenizer=tokenizer,
    model=model,
    device=device
):
    with torch.no_grad():
        inputs = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def format_retrieved_context(context_list):
    snippets = []
    for doc in context_list or []:
        snippet = (
            f"Title: {doc.get('title','')}\n"
            f"Desc: {doc.get('description','')}\n"
            f"Markdown: {doc.get('markdown','')}"
        )
        snippets.append(snippet)
    return "\n\n".join(snippets)

def format_summarized_context(summaries):
    return "\n\n".join(
        f"Title: {item['title']}\nSummary: {item['summary']}\nTactics: {', '.join(item['tactic_ids'])}"
        for item in summaries
    )

def dedup_worker_outputs(outputs, worker_type):
    """
    Deduplicate a list of worker outputs by (worker_type, worker_id).
    Only keeps one output per worker of the given type.
    """
    seen = set()
    deduped = []
    for out in outputs:
        if out.get("worker_type") != worker_type:
            continue
        wid = out.get("worker_id")
        key = (worker_type, wid)
        if key not in seen:
            deduped.append(out)
            seen.add(key)
    return deduped
