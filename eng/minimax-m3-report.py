#!/usr/bin/env python3
"""Compare TensorSharp and llama.cpp MiniMax-M3.1-Flash results from eng/minimax-m3-bench.py.

  python3 eng/minimax-m3-report.py --ts ts.jsonl --llama llama.jsonl \\
      [--ts-label layer-split] [--llama-label layer-split] \\
      [--pair "np8=ts-np8.jsonl;llama-np8.jsonl" ...] \\
      [--spec "TensorSharp DSpark=ts-dspark.jsonl;ts-plain.jsonl" ...] \\
      [--title "8x A40, UD-Q4_K_XL"] --out report.md [--html report.html]

Inputs are bench JSONL files; a comma list pools several (e.g. the A and B rounds
of an ABBA rotation). --ts-label/--llama-label keep only runs with that --label.
Every number is the median over the pooled repeats of a scenario. "TS/llama" is
the raw ratio; "TS advantage" turns it so above 1.00x means TensorSharp is better
(higher throughput, lower latency). Quality is the share of passing repeats and
of passing individual checks. --spec compares one engine with and without the
DSpark draft: decode speed-up, acceptance, and whether greedy outputs match
(only meaningful when both arms ran with the same --nonce-seed).
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path


def load(paths, label=None):
    runs, cases = {}, []
    for path in [p for p in paths.split(",") if p]:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if label is not None and rec.get("label") != label:
                continue
            if rec.get("kind") == "run":
                runs[rec["run_id"]] = rec
            elif rec.get("kind") == "case":
                cases.append(rec)
    by = {}
    for c in cases:
        by.setdefault(c["scenario"], []).append(c)
    return {"runs": runs, "cases": by}


def med(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(values) if values else None


def flat(values):
    out = []
    for v in values:
        out.extend(v if isinstance(v, list) else [v])
    return out


def fmt(v, digits=2):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:,.0f}" if digits == 0 else f"{v:,.{digits}f}"


def ratios(ts, ll, higher_is_better):
    if not isinstance(ts, (int, float)) or not isinstance(ll, (int, float)) or ts == 0 or ll == 0 or higher_is_better is None:
        return "", ""
    raw = ts / ll
    adv = raw if higher_is_better else 1 / raw
    return f"{raw:.2f}", (f"**{adv:.2f}x**" if adv >= 1 else f"{adv:.2f}x")


def metric(cases, key, sub=None):
    """Median over repeats of metrics[key] (lists are pooled; `sub` indexes per-turn lists)."""
    vals = []
    for c in cases or []:
        if c.get("verdict") == "ERROR":
            continue  # no answer came back, so its elapsed time is not a latency
        v = (c.get("metrics") or {}).get(key)
        if sub is not None and isinstance(v, list):
            v = v[sub] if isinstance(sub, int) and len(v) > sub else (v[sub] if isinstance(sub, slice) else None)
        vals.append(v)
    return med(flat(vals))


# (scenario predicate, metric label, metrics key, higher_is_better, digits)
PERF = {
    "short_chat": [("TTFT s", "ttft_s", False, 2), ("decode tok/s", "decode_tok_s", True, 2),
                   ("reasoning tokens / repeat", "reasoning_tokens", None, 0)],
    "long_prompt": [("prompt tokens", "prompt_tokens", None, 0), ("TTFT s", "ttft_s", False, 2),
                    ("prefill tok/s", "prefill_tok_s", True, 1), ("decode tok/s", "decode_tok_s", True, 2)],
    "json": [("end-to-end s", "total_s", False, 2), ("decode tok/s", "decode_tok_s", True, 2),
             ("reasoning tokens", "reasoning_tokens", None, 0)],
    "tool": [("end-to-end s", "total_s", False, 2), ("decode tok/s", "decode_tok_s", True, 2),
             ("reasoning tokens", "reasoning_tokens", None, 0)],
    "agentic_loop": [("end-to-end s", "end_to_end_s", False, 2), ("turns", "turns", None, 1),
                     ("median turn TTFT s", "ttft_s", False, 2), ("decode tok/s", "decode_tok_s", True, 2)],
    "parallel": [("aggregate tok/s", "aggregate_tok_s", True, 2), ("aggregate decode tok/s", "aggregate_decode_tok_s", True, 2),
                 ("per-request decode tok/s", "per_request_decode_tok_s", True, 2), ("TTFT s", "ttft_s", False, 2),
                 ("wall s", "wall_s", False, 2)],
    "kv_reuse": [("turn-1 TTFT s", "turn1_ttft_s", False, 2), ("turn-2/3 TTFT s", "later_ttft_s", False, 2),
                 ("TTFT drop (turn 1 / later)", "ttft_drop", True, 2),
                 ("turn-2/3 cached tokens", ("turn_cached_tokens", slice(1, None)), None, 0),
                 ("turn-2/3 cached fraction", "later_cached_fraction", None, 2)],
    "image": [("TTFT s", "ttft_s", False, 2), ("end-to-end s", "total_s", False, 2), ("decode tok/s", "decode_tok_s", True, 2)],
    "speculative": [("decode tok/s", "decode_tok_s", True, 2), ("draft acceptance", "acceptance", None, 2)],
}


def family(name):
    for prefix in ("long_prompt", "parallel", "json", "tool"):
        if name.startswith(prefix):
            return prefix
    return name


def order(names):
    rank = ["short_chat", "long_prompt_4k", "long_prompt_16k", "long_prompt_32k", "long_prompt_64k", "json_object",
            "json_schema", "tool_call_single", "tool_call_parallel", "agentic_loop", "parallel_2", "parallel_4",
            "parallel_8", "kv_reuse", "image", "speculative"]
    return sorted(names, key=lambda n: (rank.index(n) if n in rank else len(rank), n))


def perf_rows(ts, ll):
    rows = []
    for name in order(set(ts["cases"]) | set(ll["cases"])):
        a, b = ts["cases"].get(name), ll["cases"].get(name)
        for label, key, hib, digits in PERF.get(family(name), []):
            sub = None
            if isinstance(key, tuple):
                key, sub = key
            va, vb = metric(a, key, sub), metric(b, key, sub)
            if va is None and vb is None:
                continue
            raw, adv = ratios(va, vb, hib)
            rows.append([name, label, fmt(va, digits), fmt(vb, digits), raw, adv])
    return ["Scenario", "Metric", "TensorSharp", "llama.cpp", "TS/llama", "TS advantage"], rows


def quality_cell(cases):
    if not cases:
        return "—"
    n_pass = sum(1 for c in cases if c.get("pass"))
    checks = [ch for c in cases for ch in c.get("checks", [])]
    errors = sum(1 for c in cases if c.get("verdict") == "ERROR")
    cell = f"{n_pass}/{len(cases)} ({sum(ch['ok'] for ch in checks)}/{len(checks)} checks)"
    return cell + (f", {errors} error" + ("s" if errors > 1 else "") if errors else "")


def quality_notes(cases):
    notes = []
    fenced = [c for c in cases or [] if (c.get("metrics") or {}).get("fenced")]
    if fenced:
        notes.append(f"{len(fenced)} fenced ```json")
    failed = {}
    for c in cases or []:
        for ch in c.get("checks", []):
            if not ch["ok"]:
                failed[ch["name"]] = failed.get(ch["name"], 0) + 1
    if failed:
        notes.append("failed: " + ", ".join(f"{k} x{v}" for k, v in sorted(failed.items())[:4]))
    return "; ".join(notes)


def quality_rows(ts, ll):
    rows = []
    for name in order(set(ts["cases"]) | set(ll["cases"])):
        a, b = ts["cases"].get(name), ll["cases"].get(name)
        rows.append([name, quality_cell(a), quality_notes(a), quality_cell(b), quality_notes(b)])
    return ["Scenario", "TensorSharp pass", "TensorSharp notes", "llama.cpp pass", "llama.cpp notes"], rows


def tokens_rows(ts, ll):
    rows = []
    for name in order(set(ts["cases"]) | set(ll["cases"])):
        a, b = ts["cases"].get(name), ll["cases"].get(name)

        def cell(cases):
            if not cases:
                return "—", "—", ""
            src = sorted({s for c in cases for r in c.get("requests", []) for s in [r.get("reasoning_tokens_source")] if s})
            return (fmt(metric(cases, "reasoning_tokens"), 0),
                    fmt(med([sum(r.get("completion_tokens") or 0 for r in c.get("requests", [])) for c in cases]), 0),
                    ",".join(src))
        ra, rb = cell(a), cell(b)
        rows.append([name, ra[0], ra[1], ra[2], rb[0], rb[1], rb[2]])
    return ["Scenario", "TS reasoning tok", "TS completion tok", "TS source",
            "llama reasoning tok", "llama completion tok", "llama source"], rows


def spec_rows(name, with_draft, without):
    a, b = with_draft["cases"].get("speculative", []), without["cases"].get("speculative", [])
    rows = []
    kinds = sorted({p["kind"] for c in a + b for p in (c.get("metrics") or {}).get("per_prompt", [])})
    seeds_a = {r.get("nonce_seed") for r in with_draft["runs"].values()}
    seeds_b = {r.get("nonce_seed") for r in without["runs"].values()}
    same_prompts = seeds_a == seeds_b and len(seeds_a) == 1
    for kind in kinds + ["all"]:
        def per(cases):
            return [p for c in cases for p in (c.get("metrics") or {}).get("per_prompt", []) if kind in ("all", p["kind"])]
        pa, pb = per(a), per(b)
        da, db = med([p["decode_tok_s"] for p in pa]), med([p["decode_tok_s"] for p in pb])
        drafted = sum(p.get("draft_n") or 0 for p in pa)
        accepted = sum(p.get("draft_n_accepted") or 0 for p in pa)
        if same_prompts:
            sha_b = {(c["repeat"], p["kind"]): p["content_sha256"] for c in b for p in c["metrics"].get("per_prompt", [])}
            pairs = [(p["content_sha256"], sha_b.get((c["repeat"], p["kind"]))) for c in a
                     for p in c["metrics"].get("per_prompt", []) if kind in ("all", p["kind"])]
            same = f"{sum(1 for x, y in pairs if y and x == y)}/{sum(1 for _, y in pairs if y)}"
        else:
            same = "n/a (different nonce seeds)"
        raw, adv = ratios(da, db, True)
        rows.append([name, kind, fmt(da), fmt(db), adv or "—", fmt(accepted / drafted if drafted else None),
                     f"{accepted}/{drafted}" if drafted else "not reported", same])
    return rows


PERF_NOTE = ("TTFT is the first streamed reasoning, content or tool-call delta. TensorSharp sends a tool call, and "
             "response_format output, as one chunk after it is parsed, while llama.cpp streams fragments: in the "
             "JSON and tool scenarios compare end-to-end time. "
             "Repeats that returned no answer (ERROR) are left out of these medians; the quality table counts them.")
QUALITY_NOTE = ("A repeat passes when every check passes. Outputs from a random fixture model are expected to fail "
                "the checks; only the plumbing (timings, parsing, no crashes) is meaningful there.")
TOKENS_NOTE = ("Source: usage = the server's completion_tokens_details; tokenize = the reasoning text re-tokenized "
               "through /tokenize; deltas = streamed reasoning chunks (about one token each).")
SPEC_NOTE = ("Acceptance comes from llama.cpp's timings (draft_n, draft_n_accepted) or from TensorSharp's "
             "'Speculative decoding stats' log lines (bench --server-log). Identical output compares greedy outputs "
             "per prompt and repeat; it needs both arms run with the same --nonce-seed.")


SPEC_HEADER = ["Comparison", "Prompt", "decode tok/s with draft", "without", "speed-up", "acceptance",
               "accepted/drafted", "identical output"]


def run_lines(data, engine):
    out = []
    for r in data["runs"].values():
        srv = r.get("server") or {}
        props = srv.get("/props") if isinstance(srv.get("/props"), dict) else {}
        out.append(f"{engine}: run {r['run_id']} label={r.get('label')!r} model={r.get('model')} started {r.get('started')} "
                   f"host {r.get('host')} load {', '.join(f'{x:.1f}' for x in r.get('loadavg', []))}"
                   + (f" slots={props.get('total_slots')} n_ctx={props.get('n_ctx')}" if props else "")
                   + f" nonce_seed={r.get('nonce_seed')} haystack={(r.get('haystack') or {}).get('sha256', '')[:12]}")
    return out


def md_table(header, rows, left=1):
    """Columns before `left` are left-aligned text, the rest right-aligned numbers."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i < left else "---:" for i in range(len(header))) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def html_table(header, rows, left=1):
    def cell(i, x):
        text = html.escape(str(x))
        text = f"<strong>{text.strip('*')}</strong>" if text.startswith("**") else text
        return f"<td class='n'>{text}</td>" if i >= left else f"<td>{text}</td>"
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(cell(i, x) for i, x in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts", required=True, help="TensorSharp bench JSONL (comma list pools files)")
    ap.add_argument("--llama", required=True, help="llama.cpp bench JSONL (comma list pools files)")
    ap.add_argument("--ts-label")
    ap.add_argument("--llama-label")
    ap.add_argument("--pair", action="append", default=[], help="extra section: 'title=ts.jsonl;llama.jsonl'")
    ap.add_argument("--spec", action="append", default=[], help="'title=with-draft.jsonl;without-draft.jsonl'")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="markdown output")
    ap.add_argument("--html", help="also write a standalone HTML page with the same tables")
    args = ap.parse_args()

    # Each section: heading, table header (None for a bullet list), rows, left-aligned leading columns, note.
    sections = []
    ts, ll = load(args.ts, args.ts_label), load(args.llama, args.llama_label)
    sections.append(("Performance", *perf_rows(ts, ll), 2, PERF_NOTE))
    sections.append(("Quality (passing repeats, passing checks)", *quality_rows(ts, ll), 5, QUALITY_NOTE))
    sections.append(("Reasoning and completion tokens (median per repeat)", *tokens_rows(ts, ll), 1, TOKENS_NOTE))
    for spec in args.pair:
        title, files = spec.split("=", 1)
        a, b = files.split(";", 1)
        pa, pb = load(a), load(b)
        sections.append((f"{title}: performance", *perf_rows(pa, pb), 2, ""))
        sections.append((f"{title}: quality", *quality_rows(pa, pb), 5, ""))
    spec_table = []
    for spec in args.spec:
        title, files = spec.split("=", 1)
        a, b = files.split(";", 1)
        spec_table += spec_rows(title, load(a), load(b))
    if spec_table:
        sections.append(("Speculative decoding (DSpark draft)", SPEC_HEADER, spec_table, 2, SPEC_NOTE))
    sections.append(("Runs", None, run_lines(ts, "TensorSharp") + run_lines(ll, "llama.cpp"), 0, ""))

    title = f"MiniMax-M3.1-Flash: TensorSharp vs llama.cpp {args.title}".rstrip()
    intro = ("Medians over each scenario's repeats. TS/llama is the raw ratio; TS advantage above 1.00x means "
             "TensorSharp is better (higher throughput or lower latency).")
    md = [f"# {title}", "", intro, ""]
    for heading, header, rows, left, note in sections:
        md += [f"## {heading}", ""] + ([note, ""] if note else [])
        md += ([md_table(header, rows, left)] if header else [f"- {line}" for line in rows]) + [""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    if args.html:
        parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
                 "<style>body{font-family:system-ui,sans-serif;margin:24px;}table{border-collapse:collapse;margin:8px 0 24px}"
                 "td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px}td.n{text-align:right}"
                 "th{background:#f3f3f3}</style></head><body>",
                 f"<h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p>"]
        for heading, header, rows, left, note in sections:
            parts.append(f"<h2>{html.escape(heading)}</h2>" + (f"<p>{html.escape(note)}</p>" if note else ""))
            parts.append(html_table(header, rows, left) if header else
                         "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in rows) + "</ul>")
        parts.append("</body></html>")
        Path(args.html).write_text("".join(parts), encoding="utf-8")

if __name__ == "__main__":
    main()
