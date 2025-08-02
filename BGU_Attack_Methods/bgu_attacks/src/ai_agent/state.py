# state.py
from src.ai_agent.schemas import CTIEntrySchema, SummarizedContextItem
from typing import TypedDict, List, Annotated, Optional
import operator

class GraphState(TypedDict):
    raw_cti_entry: dict | None
    cti_entry: CTIEntrySchema | None
    fetched_article: str | None
    worker_outputs: Annotated[List[dict], operator.add]
    embedding: List[float] | None
    retrieved_context: List[dict] | None
    summarized_retrieved_context: Annotated[list, operator.add]
    retrieval_worker_outputs: Annotated[List[dict], operator.add]
    aggregated: Optional[bool]
    all_worker_outputs: List[dict]
    validator_outputs: List[dict]
    final_output: dict | None
    out: dict | None

class DirectWorkerState(TypedDict):
    worker_id: int
    fetched_article: str | None
    markdown: str
    worker_outputs: Annotated[List[dict], operator.add]


class RetrievalWorkerState(TypedDict):
    worker_id: int
    fetched_article: str | None
    markdown: str
    summarized_retrieved_context: List[dict] | None
    retrieval_worker_outputs: Annotated[List[dict], operator.add]

class SummarizerState(TypedDict):
    context: str
    summarized_retrieved_context: Annotated[list, operator.add]
