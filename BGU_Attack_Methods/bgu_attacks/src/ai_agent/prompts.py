from langchain_core.prompts import ChatPromptTemplate

fetch_article_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for cybersecurity research. When given a URL, use the provided tools to fetch the full article content and return it as text."),
    ("human", "Fetch and return the article at this URL: {url}")
])

direct_worker_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert cyber threat intelligence analyst.
Given the markdown and fetched article for a Sigma rule, your task is:
- Classify the Sigma rule using only MITRE tactics from the whitelist below.
- Output must be valid JSON matching:
{{
  "mitre_tags": [<one or more tactics from whitelist>],
  "reasoning": "<your reasoning, citing relevant evidence from the markdown and article>"
}}

MITRE tactics whitelist (use only these exact strings, case-sensitive):
["attack.initial_access","attack.execution","attack.persistence","attack.privilege_escalation","attack.defense_evasion","attack.credential_access","attack.discovery","attack.lateral_movement","attack.collection","attack.command_and_control","attack.exfiltration","attack.impact","attack.resource_development","attack.reconnaissance"]

INSTRUCTIONS:
- Only use the Sigma rule's markdown and fetched article for your analysis.
- Do NOT invent, modify, pluralize, or repeat tactics.
- Do NOT use any value not exactly as shown in the whitelist above.
- Do NOT output any text outside the JSON block.
- If the Sigma rule does not clearly match any tactic in the whitelist, return an empty array for "mitre_tags".
- Double check the spelling and format against the whitelist.

Example:
{{
  "mitre_tags": ["attack.persistence", "attack.execution"],
  "reasoning": "The rule detects a scheduled task creation and command-line execution."
}}
"""),
    ("human",
     "Analyze the following:\nMarkdown:\n{markdown}\n\nFetched Article:\n{fetched_article}")
])


retrieval_worker_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert cyber threat intelligence analyst.
Given:
- The markdown and fetched article for a Sigma rule
- Retrieved context: a set of similar articles from the training set

Your task:
- Classify the input Sigma rule using only MITRE tactics from the whitelist below.
- The classification must be based on the Sigma rule's markdown and fetched article. Use the retrieved context only as supporting evidence if needed, NOT as the primary basis for classification.
- Explain your reasoning, referencing relevant evidence from the markdown, article, and retrieved context.
- Output only valid JSON, matching this schema:

{{
  "mitre_tags": [<one or more tactics from whitelist>],
  "reasoning": "<clear reasoning citing sources in the provided material and context>"
}}

MITRE tactics whitelist (use only these exact strings, case-sensitive):
["attack.initial_access","attack.execution","attack.persistence","attack.privilege_escalation","attack.defense_evasion","attack.credential_access","attack.discovery","attack.lateral_movement","attack.collection","attack.command_and_control","attack.exfiltration","attack.impact","attack.resource_development","attack.reconnaissance"]

INSTRUCTIONS:
- Your classification should NOT be based solely on the retrieved context. If the Sigma rule itself does not match any tactic, do not assign tactics just because they appear in the retrieved context.
- Do not invent, modify, repeat, or pluralize tactics.
- Do not use any value not exactly as shown in the whitelist above.
- Do not output any text outside the JSON block.
- Double check the spelling and format against the whitelist.

Example:
{{
  "mitre_tags": ["attack.persistence", "attack.execution"],
  "reasoning": "The rule detects a scheduled task creation and command-line execution."
}}
"""),
    ("human",
     "Analyze the following Sigma rule and retrieved context.\nMarkdown:\n{markdown}\n\nFetched Article:\n{fetched_article}\n\nRetrieved Context:\n{summarized_retrieved_context}")
])

retrieved_context_summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a cybersecurity threat analyst.
Summarize the following Sigma rule's title, description, and markdown for a threat analyst who will use this as retrieval context.
- Focus on threat actions, MITRE tactics, detection logic, and any behaviors relevant to cyberattacks.
- Ignore installation, setup, licensing, or unrelated details.
- Write 2-3 sentences, clear and to the point.
- Do not repeat the title or description verbatim unless it's essential.
- Do not output any code, only summary text.
Return only the summary."""),
    ("human",
    "Title: {title}\nDescription: {description}\nMarkdown: {markdown}")
])