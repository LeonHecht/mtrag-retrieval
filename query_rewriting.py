import os
import json
import time
from typing import Any, Dict, List, Optional
import tqdm

from config import STORAGE_DIR, API_KEY

import cohere

COHERE_API_KEY = os.getenv("API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Set COHERE_API_KEY in your environment.")

co = cohere.ClientV2(api_key=COHERE_API_KEY)

SYSTEM_PROMPT = """You are a query-rewriting model for information retrieval.

Your task is to rewrite the FINAL user message into a SINGLE, standalone search query
that can be used by a retrieval model (BM25, dense retriever, or hybrid).

You are given the full conversation history for context.

Rules:
- Output ONLY the rewritten query text.
- Do NOT answer the question.
- Do NOT add information not implied by the conversation.
- Resolve pronouns and vague references.
- Include essential entities and technical terms.
- Remove conversational filler.
- Keep it short and retrieval-optimized.

The output must be a single sentence or compact phrase suitable for a search or embedding model.
"""

def normalize_speaker(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ["user", "human"]:
        return "user"
    if s in ["agent", "assistant", "model"]:
        return "assistant"
    return s or "unknown"

def extract_turns(record: Dict[str, Any]) -> List[Dict[str, str]]:
    turns = record.get("input", [])
    out = []
    for t in turns:
        speaker = normalize_speaker(t.get("speaker", ""))
        text = (t.get("text") or "").strip()
        if not text:
            continue
        out.append({"speaker": speaker, "text": text})
    return out

def last_user_message(turns: List[Dict[str, str]]) -> Optional[str]:
    for t in reversed(turns):
        if t["speaker"] == "user":
            return t["text"]
    return None

def context_for_rewrite(turns: List[Dict[str, str]], max_chars: int = 3500) -> str:
    """
    Build a compact transcript for the LLM.
    Keeps the whole thread but truncates from the front if too long.
    """
    lines = []
    for t in turns:
        role = "user" if t["speaker"] == "user" else "assistant"
        lines.append(f"{role}: {t['text']}")
    transcript = "\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript
    # truncate from the beginning, keep the most recent context
    return transcript[-max_chars:]

def rewrite_query_with_cohere(transcript: str, final_user_msg: str, model: str = "command-r-plus") -> str:
    prompt = f"""Conversation context:
{transcript}

Final user message:
{final_user_msg}

Rewrite the final user message into a standalone retrieval query:"""

    # Cohere Chat endpoint (ClientV2)
    resp = co.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # temperature=0.2,
    )

    # ClientV2 response content is typically in resp.message.content (list of segments)
    # We'll robustly extract text.
    text_parts = []
    msg = getattr(resp, "message", None)
    if msg and getattr(msg, "content", None):
        for seg in msg.content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                text_parts.append(seg.get("text", ""))
            elif hasattr(seg, "text"):
                text_parts.append(seg.text)
            elif isinstance(seg, str):
                text_parts.append(seg)
    rewritten = "".join(text_parts).strip()

    # Safety: ensure single-line output for jsonl
    rewritten = " ".join(rewritten.split())
    return rewritten

def process_jsonl(in_path: str, out_path: str, model: str = "command-r-plus", sleep_s: float = 0.1):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=282):

            print(f"{i}/1181")
            
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            turns = extract_turns(record)

            final_user = last_user_message(turns)
            if not final_user:
                continue

            transcript = context_for_rewrite(turns)

            rewritten = rewrite_query_with_cohere(
                transcript=transcript,
                final_user_msg=final_user,
                model=model,
            )

            # Use task_id as the output _id (matches your example style)
            out_obj = {
                "_id": record.get("task_id", record.get("conversation_id", f"line{i}")),
                "text": f"|user|: {rewritten}",
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if sleep_s:
                time.sleep(sleep_s)

if __name__ == "__main__":
    in_path = os.path.join(STORAGE_DIR, "mtrag", "data", "conversations", "conversations.jsonl")
    out_path = os.path.join(STORAGE_DIR, "mtrag", "data", "questions", "queries_rewrite_cohere.jsonl")
    
    process_jsonl(in_path, out_path, model="command-r7b-12-2024")
