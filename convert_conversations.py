#!/usr/bin/env python3
"""
Convert a JSON file containing a list of conversation objects (with `messages`)
into the MT-RAG 2.0 JSONL format (one line per user turn).

Edits vs CLI version:
- All configuration is set directly in the CONFIG dict below (no argparse / CLI flags).

Input (JSON):
[
  { "author": "...", "retriever": {...}, "generator": {...}, "messages": [...] },
  ...
]

Output (JSONL):
{"conversation_id":"...","task_id":"...<::>1","task_type":"rag","turn":"1","dataset":"...","input":[...],"Collection":"..."}
{"conversation_id":"...","task_id":"...<::>2","task_type":"rag","turn":"2","dataset":"...","input":[...],"Collection":"..."}
...
"""

import json
import sys
import uuid
from typing import Any, Dict, List, Optional
import os

from config import STORAGE_DIR

# =========================
# CONFIG (EDIT THESE)
# =========================
CONFIG = {
    # Files
    "INPUT_JSON_PATH": os.path.join(STORAGE_DIR, "mtrag", "data", "conversations", "conversations.json"),
    "OUTPUT_JSONL_PATH": os.path.join(STORAGE_DIR, "mtrag", "data", "conversations", "conversations.jsonl"),

    # Output fields
    "DATASET_NAME": "MT-RAG 2.0 Authors (Internal)",

    # conversation_id behavior:
    # - "uuid": random per conversation
    # - "author": reuse src_obj["author"] if present, else uuid
    "CONVERSATION_ID_SOURCE": "uuid",  # "uuid" or "author"

    # Collection field behavior (currently both try retriever.collection.name)
    "COLLECTION_SOURCE": "retriever",  # "retriever" or "fallback"

    # Emission behavior:
    # - "per_user_turn": write one JSONL line for each user message (snapshot)
    # - "final_only": write only the last user-turn snapshot per conversation
    "EMIT": "per_user_turn",  # "per_user_turn" or "final_only"

    # Skip messages whose text is empty/whitespace
    "DROP_EMPTY_TEXT": True,
}
# =========================


def norm_speaker(s: str) -> str:
    """Normalize speakers to exactly 'user' or 'agent' where possible."""
    s = (s or "").strip().lower()
    if s in {"assistant", "model", "agent"}:
        return "agent"
    if s in {"human", "user"}:
        return "user"
    return s or "unknown"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def make_conversation_id(src_obj: Dict[str, Any], mode: str) -> str:
    if mode == "author":
        author = src_obj.get("author")
        if author:
            return str(author)
        return uuid.uuid4().hex
    return uuid.uuid4().hex


def extract_collection(src_obj: Dict[str, Any], mode: str) -> Optional[str]:
    # Currently both modes do the same; kept for easy future tweaks.
    if mode in {"retriever", "fallback"}:
        return safe_get(src_obj, ["retriever", "collection", "name"])
    return safe_get(src_obj, ["retriever", "collection", "name"])


def build_input_entry(
    speaker: str,
    text: str,
    created_at: Optional[int],
    user_author_id: Optional[str],
    model_author_id: Optional[str],
) -> Dict[str, Any]:
    speaker_norm = norm_speaker(speaker)
    author_type = "human" if speaker_norm == "user" else "model"
    author_id = user_author_id if speaker_norm == "user" else (model_author_id or "unknown-model")

    return {
        "speaker": speaker_norm,
        "text": text,
        "metadata": {
            "author_type": author_type,
            "author_id": author_id,
            "created_at": created_at,
        },
    }


def convert_one_conversation(
    src_obj: Dict[str, Any],
    dataset_name: str,
    conversation_id_mode: str,
    collection_mode: str,
    emit: str,
    drop_empty_text: bool,
) -> List[Dict[str, Any]]:
    """
    emit:
      - "per_user_turn": emit one JSONL row per user message (snapshot up to that point)
      - "final_only": emit only the last user turn snapshot
    """
    conversation_id = make_conversation_id(src_obj, conversation_id_mode)
    collection = extract_collection(src_obj, collection_mode)

    user_author_id = src_obj.get("author")
    model_author_id = safe_get(src_obj, ["generator", "id"]) or safe_get(src_obj, ["generator", "name"])

    messages = src_obj.get("messages", [])
    if not isinstance(messages, list):
        return []

    input_so_far: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    user_turn = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        speaker = msg.get("speaker", "")
        text = (msg.get("text") or "").strip()
        if drop_empty_text and not text:
            continue

        created_at = msg.get("timestamp")

        input_so_far.append(
            build_input_entry(
                speaker=speaker,
                text=text,
                created_at=created_at,
                user_author_id=user_author_id,
                model_author_id=model_author_id,
            )
        )

        if norm_speaker(speaker) == "user":
            user_turn += 1
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "task_id": f"{conversation_id}<::>{user_turn}",
                    "task_type": "rag",
                    "turn": str(user_turn),
                    "dataset": dataset_name,
                    "input": list(input_so_far),  # snapshot
                    "Collection": collection,
                }
            )

    if emit == "final_only":
        return rows[-1:] if rows else []
    return rows


def main():
    in_path = CONFIG["INPUT_JSON_PATH"]
    out_path = CONFIG["OUTPUT_JSONL_PATH"]

    # Load input JSON
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: failed to read/parse JSON from {in_path}: {e}", file=sys.stderr)
        raise

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of conversation objects.")

    # Convert + write JSONL
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for src_obj in data:
            if not isinstance(src_obj, dict):
                continue

            rows = convert_one_conversation(
                src_obj=src_obj,
                dataset_name=CONFIG["DATASET_NAME"],
                conversation_id_mode=CONFIG["CONVERSATION_ID_SOURCE"],
                collection_mode=CONFIG["COLLECTION_SOURCE"],
                emit=CONFIG["EMIT"],
                drop_empty_text=CONFIG["DROP_EMPTY_TEXT"],
            )

            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Done. Wrote {written} JSONL lines to {out_path}.")


if __name__ == "__main__":
    main()
