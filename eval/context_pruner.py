"""Context Pruner for Gemma 4 Autonomous Developer Agent.

Implements deterministic, zero-LLM context pruning and semantic fact extraction
to prevent context window blowouts, reduce token burn, and eliminate `<|tool_call>` truncation:

1. Temporal Horizon Separation (Recency Windowing):
   - System prompt and initial user problem statement are ALWAYS preserved verbatim.
   - The most recent k turns (e.g. k=2) are preserved 100% verbatim (exact code, line numbers).
   - Historical turns (t < T - k) have their tool outputs deterministically compacted into structured facts.

2. Deterministic Semantic Fact Extraction (No Secondary LLM Required):
   - Extracts key AST symbols (class/def) from older `read_file` outputs.
   - Extracts exit status and failure loci (e.g. `AssertionError`, file/line) from older `run_command` outputs.
   - Extracts query targets and neighbor symbols from graph JSON outputs.
   - Retains exact structural facts without hallucinations, API costs, or latency overhead.

3. Head-Tail Preservation:
   - For unstructured text, preserves the informative top lines and terminal summary.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, List, Dict


def estimate_tokens(text: str) -> int:
    """Fast, conservative approximation of token count (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_semantic_tool_facts(content: str, max_chars: int = 350) -> str:
    """Deterministically extracts key facts from older tool outputs without calling any LLM.
    
    Handles:
    1. Python code reads (extracts function/class signatures and total line count)
    2. Test & terminal execution (extracts exit status, failing test names, and assertion locus)
    3. Graph tool JSON responses (extracts queried symbols and top neighbor names)
    4. Unstructured text (head-tail preservation)
    """
    if not content or len(content) <= max_chars:
        return content

    # --- Heuristic 1: Graph Tool JSON Responses ---
    if content.strip().startswith("{") and content.strip().endswith("}"):
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                status = data.get("status", "ok")
                if "neighbors" in data:
                    symbols = [n.get("target") or n.get("node") for n in data["neighbors"][:5]]
                    return f"[GRAPH NEIGHBORS COMPACTED] Query neighbors found: {symbols}. Total: {len(data['neighbors'])}. (JSON compacted)."
                if "results" in data:
                    hits = [r.get("symbol") or r.get("filepath") for r in data["results"][:4]]
                    return f"[SIMILAR CODE SEARCH COMPACTED] Hits: {hits}. Total matches: {len(data['results'])}. (JSON compacted)."
                if "nodes" in data:
                    return f"[CODE SUBGRAPH COMPACTED] Subgraph with {data.get('node_count', len(data['nodes']))} nodes, {data.get('edge_count', 0)} edges. (JSON compacted)."
        except Exception:
            pass

    # --- Heuristic 2: Pytest and Terminal Command Execution ---
    if "pytest" in content or "FAILED" in content or "PASSED" in content or "Traceback" in content:
        # Extract failure assertions and summary
        errors = re.findall(r'((?:AssertionError|SyntaxError|AttributeError|TypeError|ValueError|FAILED\s+[^\n]+)[^\n]*)', content)
        summary_line = re.search(r'(=+\s*[\d\w\s,]+in\s+[\d\.]+s\s*=+|exit\s+code\s+\d+)', content, re.IGNORECASE)
        
        status = "PASSED" if ("PASSED" in content and "FAILED" not in content and "ERROR" not in content) else "FAILED"
        summary_str = summary_line.group(1) if summary_line else f"Status: {status}"
        
        if errors:
            key_errors = "; ".join(errors[:2])
            return f"[TEST EXECUTION COMPACTED] {summary_str}. Error locus: {key_errors}. (Full logs compacted)."
        return f"[TEST EXECUTION COMPACTED] {summary_str}. (Logs compacted)."

    # --- Heuristic 3: Python Source Code File Reads ---
    symbols = re.findall(r'^\s*(?:def|class|async def)\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
    if symbols:
        line_count = len(content.splitlines())
        unique_symbols = list(dict.fromkeys(symbols))[:6]
        return (
            f"[FILE CONTENT COMPACTED] Inspected {line_count} lines. "
            f"Key symbols defined: {unique_symbols}. "
            f"(Source code lines compacted to preserve context; call read_file on targeted lines if needed)."
        )

    # --- Heuristic 4: Unstructured Text (Head-Tail Preservation) ---
    half = max_chars // 2
    head = content[:half].rstrip()
    tail = content[-half:].lstrip()
    omitted = len(content) - max_chars
    return f"{head}\n\n... [PRUNED: {omitted} characters omitted for brevity] ...\n\n{tail}"


def prune_tool_output(content: str, max_chars: int = 1200) -> str:
    """Truncates excessive tool outputs while preserving head and tail."""
    if len(content) <= max_chars:
        return content
    
    half = max_chars // 2
    head = content[:half]
    tail = content[-half:]
    omitted = len(content) - max_chars
    return f"{head}\n\n... [PRUNED: {omitted} characters / {omitted // 4} tokens omitted for brevity] ...\n\n{tail}"


def prune_conversation_history(
    messages: list[dict[str, Any]],
    max_history_tokens: int = 6000,
    keep_recent_turns: int = 2,
) -> list[dict[str, Any]]:
    """Prunes older tool responses in the conversation while preserving system prompt and recent turns.
    
    Architecture:
    - messages[0] is system prompt (ALWAYS preserved verbatim).
    - messages[1] is the initial user problem statement (ALWAYS preserved verbatim).
    - The most recent `keep_recent_turns` turns are kept 100% verbatim.
    - Intermediate tool outputs in older turns are compacted into structured facts using deterministic extraction.
    """
    # Guard: if history is small, return untouched
    if len(messages) <= 2 + (keep_recent_turns * 2):
        return messages

    pruned_messages = []
    # Always preserve system prompt and problem statement
    pruned_messages.append(messages[0])
    pruned_messages.append(messages[1])

    # Slice into historical middle vs recent turns
    cutoff_index = len(messages) - (keep_recent_turns * 2)
    middle_messages = messages[2:cutoff_index]
    recent_messages = messages[cutoff_index:]

    for msg in middle_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        # If it's a tool output message in an older turn, deterministically extract facts
        if role == "tool":
            compacted_content = extract_semantic_tool_facts(content, max_chars=350)
            pruned_msg = dict(msg)
            pruned_msg["content"] = compacted_content
            pruned_messages.append(pruned_msg)
        elif role == "assistant" and msg.get("tool_calls"):
            # Assistant tool calls are kept but lengthy thought scratchpads are trimmed
            pruned_msg = dict(msg)
            if content and len(content) > 250:
                pruned_msg["content"] = content[:250] + " ... [Reasoning compacted]"
            pruned_messages.append(pruned_msg)
        else:
            pruned_messages.append(msg)

    # Append recent turns completely untouched
    pruned_messages.extend(recent_messages)
    return pruned_messages


def slice_ast_symbol(code: str, target_symbol: str) -> str | None:
    """Extracts only the specified class or function definition from a Python file using AST parsing."""
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    lines = code.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == target_symbol:
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", len(lines))
                return "".join(lines[start_line:end_line])

    return None
