"""Regenerate independent Jinja2 prompt expectations. Requires jinja2 3.1.6."""
import json
from pathlib import Path
import re
import jinja2

root = Path(__file__).resolve().parents[2]
fixtures = root / "InferenceWeb.Tests" / "Fixtures" / "MuseSpark"
cases = []
for template_name in ("chat_template.jinja", "unsloth_chat_template.jinja"):
    source = (fixtures / template_name).read_text(encoding="utf-8")
    # HF's generation extension only annotates token spans, without changing text.
    source = re.sub(r"\{%[-+]?\s*(?:end)?generation\s*[-+]?%\}", "", source)
    environment = jinja2.Environment()
    # llama.cpp/minja and TensorSharp retain caller schema property order.
    environment.policies["json.dumps_kwargs"] = {"sort_keys": False, "ensure_ascii": False}
    template = environment.from_string(source)
    contexts = [
        {"messages": [{"role": "user", "content": "Hi <|eot|> <image>."}]},
        {"messages": [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "Describe this."}]}],
         "reasoning_effort": "low"},
        {"messages": [{"role": "user", "content": "Weather?"}],
         "tools": [{"type": "function", "function": {"name": "weather.get", "description": "Get weather.",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]},
    ]
    for index, context in enumerate(contexts):
        context.update(bos_token="", add_generation_prompt=True)
        cases.append({"template": template_name, "case": index, "context": context,
                      "expected": template.render(**context)})
(fixtures / "jinja_reference.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(cases)} reference prompts with Jinja2 {jinja2.__version__}.")
