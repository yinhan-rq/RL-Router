#!/usr/bin/env python3
import argparse, csv, os, re, sys, time
from typing import Iterable, Tuple, Optional

# pip install openai>=1.30
from openai import OpenAI


PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

def read_pairs(path: str) -> Iterable[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = PAIR_RE.search(line)
            if m:
                yield int(m.group(1)), int(m.group(2))

def make_prompt(n_tokens_hint: int) -> str:
    # Simple token-length hint: repeat a short word with spaces
    # (not exact token counts, but adequate for quick latency tests)
    return ("word " * max(1, n_tokens_hint)).strip()

def run_one(client: OpenAI, model: str, in_len: int, out_len: int, timeout: float):
    prompt = make_prompt(in_len)

    # Start timing and stream the response
    start = time.perf_counter()
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=out_len,
        temperature=0.0,
        stream=True,
        timeout=timeout,
    )

    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    est_completion_tokens = 0
    finish_reason = ""

    for chunk in stream:
        now = time.perf_counter()
        choice = chunk.choices[0]

        # Treat any non-empty text as a token arrival
        text_piece = getattr(choice, "text", "") or ""
        if text_piece:
            est_completion_tokens += 1
            if first_token_time is None:
                first_token_time = now

        # Final chunk has a finish_reason set (e.g., "length", "stop")
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason or ""
            last_token_time = now

    # Safety fallback if final flag was missing
    if last_token_time is None:
        last_token_time = time.perf_counter()

    # If model streamed empty first piece (rare), fallback to first event time
    if first_token_time is None:
        first_token_time = last_token_time

    ttft_ms = (first_token_time - start) * 1000.0
    decode_ms = max(0.0, (last_token_time - first_token_time) * 1000.0)
    total_ms = (last_token_time - start) * 1000.0
    tok_per_s = (est_completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0

    return {
        "input_len_hint": in_len,
        "output_len_req": out_len,
        "ttft_ms": round(ttft_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "total_ms": round(total_ms, 3),
        "est_completion_tokens": est_completion_tokens,
        "decode_tok_per_s": round(tok_per_s, 3),
        "finish_reason": finish_reason,
    }

def main():
    ap = argparse.ArgumentParser(description="Simple vLLM latency benchmark (TTFT & decode).")
    ap.add_argument("--pairs", required=True, help="Path to text file of pairs like '(928, 492)' per line")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                    help="OpenAI-compatible base URL (default: http://127.0.0.1:8000/v1)")
    ap.add_argument("--model", required=True, help="Model name as served by vLLM (e.g., meta-llama/Llama-3-8B-Instruct)")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"),
                    help="API key (vLLM accepts any non-empty string; default: 'EMPTY')")
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (seconds)")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    rows = []
    for i, (inp, outp) in enumerate(read_pairs(args.pairs), 1):
        try:
            res = run_one(client, args.model, inp, outp, args.timeout)
            rows.append(res)
            print(f"[{i}] in={inp} out={outp}  TTFT={res['ttft_ms']} ms  "
                  f"Decode={res['decode_ms']} ms  toks/s={res['decode_tok_per_s']}")
        except Exception as e:
            print(f"[{i}] in={inp} out={outp}  ERROR: {e}", file=sys.stderr)
            rows.append({
                "input_len_hint": inp,
                "output_len_req": outp,
                "ttft_ms": "", "decode_ms": "", "total_ms": "",
                "est_completion_tokens": "", "decode_tok_per_s": "",
                "finish_reason": f"ERROR: {e}",
            })

    fieldnames = ["input_len_hint", "output_len_req", "ttft_ms", "decode_ms",
                  "total_ms", "est_completion_tokens", "decode_tok_per_s", "finish_reason"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()