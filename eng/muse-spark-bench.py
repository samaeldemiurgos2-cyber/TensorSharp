#!/usr/bin/env python3
"""Throughput / latency benchmark for an OpenAI-compatible Muse Spark server.

Runs the SAME requests against TensorSharp or llama.cpp and records client-side
timings from the SSE stream (time to first token, decode rate) plus whatever
the server reports (usage, cached tokens, llama.cpp `timings`). It never starts
servers; launch each engine with matched settings and point --url at it.

  python3 eng/muse-spark-bench.py --engine llamacpp --url http://127.0.0.1:8082 \
      --out artifacts/muse-spark-bench/llama.json
  python3 eng/muse-spark-bench.py --engine tensorsharp --url http://127.0.0.1:8083 \
      --out artifacts/muse-spark-bench/tensorsharp.json

Scenarios (comma list via --scenarios):
  short        ~160-token prompt, --decode-tokens generated (decode speed)
  long_4k      ~4K-token prompt, 32 generated (prefill speed)
  long_16k     ~16K-token prompt, 32 generated
  parallel_N   N concurrent short requests (aggregate decode throughput)
  cache_reuse  two-turn conversation over a ~3K shared prefix (second-turn TTFT)
  shared_prefix four new conversations sharing one ~2.3K system prompt (agent pattern)

Short and parallel prompts start with a request number that differs per repeat
(and is the same for every engine), so a repeat measures a cold prefill:
llama.cpp reuses any cached prefix across requests, while TensorSharp shares only
public (system-prompt) prefixes between unrelated conversations. Prefix reuse is
measured by cache_reuse and shared_prefix. A random "run id" prefix is avoided:
the model refused some prompts that carried one.

Every request is greedy (temperature 0) with thinking disabled. Decode rate is
(completion_tokens - 1) / (t_last_token - t_first_token): it excludes prefill
and is comparable across engines because both stream token by token.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

FILLER = (
    "The archive of the northern observatory records weather, tides and the positions of the planets. "
    "Each entry lists the date, the observer, the instrument used and any corrections applied later. "
    "Most entries are routine, but some describe storms, eclipses or equipment failures in detail. "
)


MODEL = "muse-spark"


def discover_model(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"][0]["id"]
    except Exception:
        return "muse-spark"


def body_for(engine, messages, max_tokens, extra=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if engine == "tensorsharp":
        body.update({"think": False, "skills": [], "skills_discovery": False, "multi_agent": False})
    else:
        # Matched reference template (eng/validation/prepare_muse_spark_llama_reference.py):
        # llama.cpp's Spark parser needs plain content parsing and an end-of-turn
        # stop for primed direct answers (see eng/muse-spark-validate.py).
        body.update({"chat_template_kwargs": {"enable_thinking": False, "current_date": ""},
                     "reasoning_format": "none", "stop": ["<|eot|>"]})
    if extra:
        body.update(extra)
    return body


def stream_chat(url, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n_events = 0
    text = []
    usage = None
    timings = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                timings = obj["timings"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece or delta.get("tool_calls"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_events += 1
                    text.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    usage = usage or {}
    completion = usage.get("completion_tokens") or n_events
    prompt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and timings:
        cached = timings.get("cache_n")
    res = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "ttft_s": (first - t0) if first else None,
        "total_s": t_end - t0,
        "decode_tok_s": ((completion - 1) / (last - first)) if first and last and last > first and completion > 1 else None,
        "finish_reason": finish,
        "text": "".join(text),
    }
    if prompt and res["ttft_s"]:
        uncached = prompt - (cached or 0)
        res["prefill_tok_s"] = uncached / res["ttft_s"] if uncached > 0 else None
    if timings:
        res["server_timings"] = timings
    return res


def long_prompt(n_words_target):
    words = []
    i = 0
    while len(words) < n_words_target:
        words.extend((f"[{i}] " + FILLER).split())
        i += 1
    doc = " ".join(words[:n_words_target])
    secret = "The access code for the eastern dome is 7314."
    mid = len(doc) // 2
    doc = doc[:mid] + " " + secret + " " + doc[mid:]
    return [{"role": "user", "content": doc + "\n\nWhat is the access code for the eastern dome? Answer with the number only."}]


SHORT_PROMPT = [{"role": "user", "content": "Write a detailed, multi-paragraph explanation of how a hash table works, "
                                            "including hashing, collisions, chaining, open addressing and resizing."}]


def run_short(args, label="short"):
    out = []
    for r in range(args.repeats):
        msgs = [{"role": "user", "content": f"Request {100 + r}: " + SHORT_PROMPT[0]["content"]}]
        res = stream_chat(args.url, body_for(args.engine, msgs, args.decode_tokens), args.timeout)
        res["repeat"] = r
        out.append(res)
        print(f"  {label} #{r}: ttft {res['ttft_s']:.2f}s decode {res['decode_tok_s'] or 0:.2f} tok/s "
              f"({res['completion_tokens']} tok, prompt {res['prompt_tokens']})", flush=True)
    return out


def run_long(args, words, label):
    out = []
    for r in range(args.repeats):
        # Vary the prompt per repeat so prompt caches cannot help.
        msgs = long_prompt(words)
        msgs[0]["content"] = f"Run {r} {time.time_ns()}.\n" + msgs[0]["content"]
        res = stream_chat(args.url, body_for(args.engine, msgs, 32, {"cache_prompt": False}), args.timeout)
        res["repeat"] = r
        res["needle_ok"] = "7314" in res["text"]
        out.append(res)
        print(f"  {label} #{r}: prompt {res['prompt_tokens']} ttft {res['ttft_s']:.2f}s "
              f"prefill {res.get('prefill_tok_s') or 0:.1f} tok/s needle={'ok' if res['needle_ok'] else 'MISS'} "
              f"answer={res['text'][:40]!r}", flush=True)
    return out


def run_parallel(args, n):
    out = []
    for r in range(args.repeats):
        results = [None] * n
        prompts = [
            [{"role": "user", "content": f"Request {200 + 10 * r + i}: " + SHORT_PROMPT[0]["content"]}]
            for i in range(n)
        ]

        def worker(i):
            results[i] = stream_chat(args.url, body_for(args.engine, prompts[i], args.decode_tokens), args.timeout)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total_tokens = sum(x["completion_tokens"] for x in results)
        rec = {
            "repeat": r, "n": n, "wall_s": wall, "total_completion_tokens": total_tokens,
            "aggregate_tok_s": total_tokens / wall,
            "per_request_decode_tok_s": [x["decode_tok_s"] for x in results],
            "ttft_s": [x["ttft_s"] for x in results],
            "requests": results,
        }
        out.append(rec)
        print(f"  parallel_{n} #{r}: wall {wall:.1f}s aggregate {rec['aggregate_tok_s']:.2f} tok/s "
              f"per-req decode {[round(x or 0, 2) for x in rec['per_request_decode_tok_s']]}", flush=True)
    return out


def run_cache_reuse(args):
    out = []
    for r in range(args.repeats):
        system = f"Session {r}-{time.time_ns()}. You are a careful assistant.\n" + " ".join((FILLER * 40).split()[:2400])
        q1 = [{"role": "system", "content": system},
              {"role": "user", "content": "In one sentence, what does the archive record?"}]
        a = stream_chat(args.url, body_for(args.engine, q1, 48), args.timeout)
        q2 = q1 + [{"role": "assistant", "content": a["text"]},
                   {"role": "user", "content": "Name one unusual kind of event the entries describe. One word."}]
        b = stream_chat(args.url, body_for(args.engine, q2, 16), args.timeout)
        rec = {"repeat": r, "turn1": a, "turn2": b}
        out.append(rec)
        print(f"  cache_reuse #{r}: turn1 prompt {a['prompt_tokens']} ttft {a['ttft_s']:.2f}s | "
              f"turn2 prompt {b['prompt_tokens']} cached {b['cached_tokens']} ttft {b['ttft_s']:.2f}s", flush=True)
    return out


def run_shared_prefix(args):
    """Separate conversations sharing one long system prompt (the agent pattern:
    instructions and tool schemas first, a different user turn each time)."""
    out = []
    for r in range(args.repeats):
        system = (f"Deployment {r}-{time.time_ns()}. You are a support agent for the northern observatory. "
                  + " ".join((FILLER * 40).split()[:2300]))
        turns = []
        for q in ["What does the archive record? One sentence.", "Name one instrument mentioned. One word.",
                  "Who writes the entries? One phrase.", "What do the entries describe besides routine? Briefly."]:
            res = stream_chat(args.url, body_for(args.engine, [{"role": "system", "content": system},
                                                              {"role": "user", "content": q}], 24), args.timeout)
            turns.append(res)
        rec = {"repeat": r, "turns": turns}
        out.append(rec)
        print(f"  shared_prefix #{r}: ttft {[round(t['ttft_s'], 2) for t in turns]} cached {[t['cached_tokens'] for t in turns]}",
              flush=True)
    return out


def summarize(results):
    def med(values):
        values = [v for v in values if isinstance(v, (int, float))]
        return statistics.median(values) if values else None
    s = {}
    for name, runs in results.items():
        if name.startswith("parallel"):
            s[name] = {"aggregate_tok_s": med([x["aggregate_tok_s"] for x in runs]),
                       "wall_s": med([x["wall_s"] for x in runs])}
        elif name == "shared_prefix":
            s[name] = {"first_ttft_s": med([x["turns"][0]["ttft_s"] for x in runs]),
                       "later_ttft_s": med([t["ttft_s"] for x in runs for t in x["turns"][1:]]),
                       "later_cached_tokens": med([t["cached_tokens"] for x in runs for t in x["turns"][1:]])}
        elif name == "cache_reuse":
            s[name] = {"turn2_ttft_s": med([x["turn2"]["ttft_s"] for x in runs]),
                       "turn1_ttft_s": med([x["turn1"]["ttft_s"] for x in runs]),
                       "turn2_cached_tokens": med([x["turn2"]["cached_tokens"] for x in runs])}
        else:
            s[name] = {"ttft_s": med([x["ttft_s"] for x in runs]),
                       "decode_tok_s": med([x["decode_tok_s"] for x in runs]),
                       "prefill_tok_s": med([x.get("prefill_tok_s") for x in runs]),
                       "prompt_tokens": med([x["prompt_tokens"] for x in runs]),
                       "completion_tokens": med([x["completion_tokens"] for x in runs])}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["tensorsharp", "llamacpp"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", default="short,long_4k,long_16k,parallel_4,cache_reuse")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    global MODEL
    MODEL = discover_model(args.url)

    if args.warmup:
        print("warmup...", flush=True)
        stream_chat(args.url, body_for(args.engine, [{"role": "user", "content": "Say hi."}], 8), args.timeout)

    results = {}
    for name in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        print(f"[{name}]", flush=True)
        if name == "short":
            results[name] = run_short(args)
        elif name.startswith("long_"):
            k = int(name[5:].rstrip("k"))
            # ~1.3 tokens per word for this filler
            results[name] = run_long(args, int(k * 1000 / 1.3), name)
        elif name.startswith("parallel_"):
            results[name] = run_parallel(args, int(name.split("_")[1]))
        elif name == "cache_reuse":
            results[name] = run_cache_reuse(args)
        elif name == "shared_prefix":
            results[name] = run_shared_prefix(args)
        else:
            print(f"unknown scenario {name}", file=sys.stderr)
            sys.exit(2)
    report = {"engine": args.engine, "url": args.url, "label": args.label, "decode_tokens": args.decode_tokens,
              "repeats": args.repeats, "summary": summarize(results), "results": results,
              "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
