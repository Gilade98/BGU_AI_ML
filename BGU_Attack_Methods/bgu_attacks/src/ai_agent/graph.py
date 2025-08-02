from src.ai_agent.state import *
from src.ai_agent.schemas import *
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langgraph.types import Send
from src.ai_agent.prompts import *
from src.ai_agent.tools import *
from collections import Counter
from src.config import *
import logging

llm = ChatOllama(model="llama3.1:8b")


def cti_entry_node(state: GraphState) -> GraphState:
    # Parse and validate with pydantic model
    state["cti_entry"] = CTIEntrySchema.parse_obj(state["raw_cti_entry"])
    return state

def fetch_article_node(state: GraphState) -> GraphState:
    chain = (
        fetch_article_prompt
        | llm.bind_tools([fetch_article]).with_structured_output(FetchArticleOutput)
    )

    url = state["cti_entry"].url
    result: FetchArticleOutput = chain.invoke({"url": url})

    # update state
    state["fetched_article"] = result.article_content if result.status == "success" else None
    logging.debug(f"Fetched article with status: {result.status}" )
    logging.debug(f"Fetched article: {state['fetched_article']}")
    return state

def assign_direct_workers(state: GraphState) -> List[Send]:
    return [Send("Direct Worker", {"worker_id": wid,
                                      "fetched_article": state["fetched_article"],
                                      "markdown": state["cti_entry"].markdown})
            for wid in range(NUM_DIRECT_WORKERS)]
def direct_worker_node(state: DirectWorkerState) -> dict:
    chain = (
        direct_worker_prompt
        | llm.with_structured_output(DirectWorkerOutput)
    )
    result: DirectWorkerOutput = chain.invoke({
        "markdown": state["markdown"],
        "fetched_article": state["fetched_article"],
    })
    logging.debug(f"Direct worker {state['worker_id']} done")
    logging.debug(f"Direct Worker {state['worker_id']} result: {result.dict()}")
    result_dict = result.dict()
    result_dict["worker_type"] = "direct"
    result_dict["worker_id"] = state["worker_id"]
    return {"worker_outputs": [result_dict]}

def embedder_node(state: GraphState) -> GraphState:
    markdown = (state["cti_entry"].markdown or "")
    article = (state["fetched_article"] or "")
    test_text = markdown + "\n" + article
    state["embedding"] = embed_text_secbert(test_text)
    logging.debug("Done Embedding")
    logging.debug(f"Embeddings: {state['embedding']}")
    return state

def retriever_node(state: GraphState) -> GraphState:
    D, I = faiss_index.search(state["embedding"], RETRIEVAL_K)
    retrieved = train_df.iloc[I[0]].to_dict(orient="records")
    # Remove 'tech_ids' and re-prefix 'tactic_ids'
    clean_records = []
    for rec in retrieved:
        # Clean tactic_ids
        tactic_ids = [
            t if t.startswith("attack.") else "attack." + t
            for t in rec.get("tactic_ids", [])
        ]
        # Only keep the desired keys
        clean_rec = {
            "title": rec.get("title", ""),
            "description": rec.get("description", ""),
            "markdown": rec.get("markdown", ""),  # optionally truncate here if it's too long!
            "tactic_ids": tactic_ids
        }
        clean_records.append(clean_rec)

    state["retrieved_context"] = clean_records
    logging.debug("Done Retrieving")
    logging.debug(f"Retrieved: {state['retrieved_context']}")
    return state

def summarize_retrieved_node(state: SummarizedContextItem):
    rec = state["context"]
    chain = (
            retrieved_context_summarizer_prompt
            | llm  # no structured output needed, just plain text
    )
    summary = chain.invoke({
        "title": rec["title"],
        "description": rec["description"],
        "markdown": rec["markdown"]
    })
    summary_content = summary.content
    if "<think>" in summary_content and "</think>" in summary_content:
        # Extract only the content after </think>
        summary_content = summary_content.split("</think>")[-1].strip()

    logging.debug(f"Summary : {summary_content}")
    logging.debug("Done Summary")
    return {"summarized_retrieved_context": [{
            "title": rec["title"],
            "summary": summary_content,
            "tactic_ids": rec["tactic_ids"],
        }]
    }

def assign_summary_workers(state: GraphState):
    return [Send("Summarize Worker", {"context": s}) for s in state["retrieved_context"]]



def retrieval_worker_node(state: RetrievalWorkerState) -> dict:
    # Format retrieved context (summarize or limit to avoid token overflow)
    context_snippets = format_summarized_context(state["summarized_retrieved_context"])
    chain = (
        retrieval_worker_prompt
        | llm.with_structured_output(RetrievalWorkerOutput)
    )
    result: RetrievalWorkerOutput = chain.invoke({
        "markdown": state["markdown"],
        "fetched_article": state["fetched_article"],
        "summarized_retrieved_context": context_snippets,
    })
    logging.debug(f"Retrieving worker: {state['worker_id']} done")
    result_dict = result.dict()
    result_dict["mitre_tags"] = [t for t in result_dict["mitre_tags"] if t in TACTIC_WHITELIST]
    result_dict["worker_type"] = "retrieval"
    result_dict["worker_id"] = state["worker_id"]
    logging.debug(f"Retrieving worker: {state['worker_id']} output: {result.dict()}")
    return {"retrieval_worker_outputs": [result_dict]}

def assign_retrieval_workers(state: GraphState):
    return [Send("Retrieval Worker", {"worker_id": wid,
                                      "fetched_article": state["fetched_article"],
                                      "markdown": state["cti_entry"].markdown,
                                      "summarized_retrieved_context": state["summarized_retrieved_context"]
                                      }) for wid in range(NUM_RETRIEVAL_WORKERS)]

def aggregator_node(state: GraphState) -> GraphState:
    direct_outputs = state.get("worker_outputs", [])
    retrieval_outputs = state.get("retrieval_worker_outputs", [])

    deduped_direct = dedup_worker_outputs(direct_outputs, "direct")
    deduped_retrieval = dedup_worker_outputs(retrieval_outputs, "retrieval")

    direct_count = len(deduped_direct)
    retrieval_count = len(deduped_retrieval)

    logging.debug(f"Aggregator barrier: {direct_count}/{NUM_DIRECT_WORKERS} direct, {retrieval_count}/{NUM_RETRIEVAL_WORKERS} retrieval")

    if state.get("aggregated", False):
        logging.debug("Aggregator: already aggregated, skipping.")
        return state

    if direct_count == NUM_DIRECT_WORKERS and retrieval_count == NUM_RETRIEVAL_WORKERS:
        logging.debug("Aggregator: all workers done, proceeding to aggregate.")
        deduped_outputs = deduped_direct + deduped_retrieval
        logging.debug("Number of outputs (deduped):", len(deduped_outputs))
        state["all_worker_outputs"] = deduped_outputs
        state["aggregated"] = True
    else:
        logging.debug("Aggregator: Not all workers done, waiting...")

    logging.debug("Done Aggregator")
    return state

def validator_node(state: GraphState) -> GraphState:

    if "all_worker_outputs" not in state:
        logging.debug("Validator: all_worker_outputs not present, waiting...")
        return state  # Don't run, just return unchanged

    deduped_outputs = state["all_worker_outputs"]

    all_tags = []
    all_reasonings = []
    for out in deduped_outputs:
        all_tags.extend(set(out.get("mitre_tags", [])))  # Deduplicate per worker
        all_reasonings.append({
            "worker_id": out.get("worker_id", None),
            "worker_type": out.get("worker_type", ""),
            "reasoning": out.get("reasoning", "")
        })
    tag_counts = Counter(all_tags)
    num_workers = len(deduped_outputs)
    threshold = 0.5

    # Multi-label voting
    final_tags = [
        tag for tag, count in tag_counts.items()
        # if count / num_workers >= threshold # at least half of the workers
    ]
    final_tags = [tag for tag in final_tags if tag in TACTIC_WHITELIST]

    filtered_tag_counts = {tag: tag_counts.get(tag, 0) for tag in TACTIC_WHITELIST}
    total_votes_in_whitelist = sum(filtered_tag_counts.values())
    tag_probs = {
        tag: (filtered_tag_counts[tag] / total_votes_in_whitelist if total_votes_in_whitelist > 0 else 0.0)
        for tag in TACTIC_WHITELIST
    }
    state["final_output"] = {
        "final_tags": final_tags,
        "tag_votes": dict(tag_counts),
        "tag_probs": tag_probs,
        "all_reasonings": all_reasonings
    }
    logging.debug("Done Validating")
    return state

def output_node(state: GraphState) -> GraphState:

    if "final_output" not in state:
        logging.debug("Output: final_output not present, waiting...")
        return state

    out = {}
    out["url"] = state["cti_entry"].url if state["cti_entry"] else None
    for tag in TACTIC_WHITELIST:
        out[f"llm_prob_{tag}"] = state["final_output"]["tag_probs"].get(tag, 0.0)
        out[f"llm_vote_{tag}"] = state["final_output"]["tag_votes"].get(tag, 0)
        out[f"llm_final_{tag}"] = 1 if tag in state["final_output"]["final_tags"] else 0
    state["out"] = out
    logging.debug("Done output")
    return state
def build_and_compile_graph():
    graph = StateGraph(GraphState)

    # Add nodes:
    graph.add_node("CTI Entry", cti_entry_node)
    graph.add_node("Fetch Article", fetch_article_node)
    graph.add_node("Embedder", embedder_node)
    graph.add_node("Retriever", retriever_node)
    graph.add_node("Summarize Worker", summarize_retrieved_node)
    graph.add_node("Retrieval Worker", retrieval_worker_node, map=True)
    graph.add_node("Assign Retrieval Workers", lambda state: {}, map=True)  # dummy, optional
    graph.add_node("Assign Direct Workers", lambda state: {}, map=True)
    graph.add_node("Direct Worker", direct_worker_node)
    graph.add_node("Aggregator", aggregator_node, defer=True)
    graph.add_node("Validator", validator_node)
    graph.add_node("Output", output_node)

    # Wire edges:
    graph.add_edge(START, "CTI Entry")

    graph.add_edge("CTI Entry", "Fetch Article")
    # Edges for direct branch
    graph.add_edge("Fetch Article", "Assign Direct Workers")  # optional dummy node
    graph.add_conditional_edges("Assign Direct Workers", assign_direct_workers, ["Direct Worker"])
    # Before aggregating, future work should add validate -> retry functionality as a conditional edge for each worker, in case of empty tags
    graph.add_edge("Direct Worker", "Aggregator")

    graph.add_edge("Fetch Article", "Embedder")
    graph.add_edge("Embedder", "Retriever")
    graph.add_conditional_edges("Retriever", assign_summary_workers, ["Summarize Worker"])
    graph.add_edge("Summarize Worker", "Assign Retrieval Workers")
    graph.add_conditional_edges("Assign Retrieval Workers", assign_retrieval_workers, ["Retrieval Worker"])
    # Before aggregating, future work should add validate -> retry functionality as a conditional edge for each worker, in case of empty tags
    graph.add_edge("Retrieval Worker", "Aggregator")

    graph.add_edge("Aggregator", "Validator")
    graph.add_edge("Validator", "Output")
    graph.add_edge("Output", END)

    compiled_graph = graph.compile()

    return compiled_graph

compiled_graph = build_and_compile_graph()
