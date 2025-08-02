from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field

class SigmaRuleSchema(BaseModel):
    title: str
    id: str
    status: str
    description: str
    references: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    date: Optional[str] = None
    modified: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    logsource: Dict[str, Any] = Field(default_factory=dict)
    detection: Dict[str, Any] = Field(default_factory=dict)
    falsepositives: List[str] = Field(default_factory=list)
    level: Optional[str] = None

class CTIEntrySchema(BaseModel):
    url: str
    markdown: str
    sigma_rule: SigmaRuleSchema
    sigma_rule_category: Optional[str] = None

class FetchArticleOutput(BaseModel):
    article_content: str
    status: str = "success"  # or "error"
    error_message: str = ""

class DirectWorkerOutput(BaseModel):
    mitre_tags: List[str]
    reasoning: str

class RetrievalWorkerOutput(BaseModel):
    mitre_tags: List[str]
    reasoning: str
class SummarizedContextItem(BaseModel):
    title: str
    summary: str
    tactic_ids: List[str]