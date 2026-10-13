"""Minimal API clients; credentials stay outside the repository."""
import json
from pathlib import Path
import os
import urllib.error
import urllib.request


CONFIG = Path(os.environ.get("LINGUAFORCE_API_CONFIG",
                           str(Path.home() / ".config/linguaforce/api_config.json")))


def configuration(provider):
    prefix = "LINGUAFORCE_" + provider.upper() + "_"
    config = json.loads(CONFIG.read_text()).get(provider, {}) if CONFIG.exists() else {}
    for field in ("base_url", "model", "api_key", "api"):
        if prefix + field.upper() in os.environ:
            config[field] = os.environ[prefix + field.upper()]
    config.setdefault("api", "responses" if provider == "gpt" else "chat_completions")
    missing = [field for field in ("base_url", "model", "api_key") if not config.get(field)]
    if missing:
        raise ValueError(f"Configure {provider}: missing {', '.join(missing)}")
    return config


def request(config, path, payload=None, timeout=90):
    req = urllib.request.Request(config["base_url"].rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + config["api_key"],
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace").replace(config["api_key"], "[REDACTED]")
        raise RuntimeError(f"HTTP {exc.code}: {body[:800]}") from None


def complete(provider, prompt, max_tokens=1200):
    config = configuration(provider)
    payload = {"model": config["model"]}
    if config.get("api") == "responses":
        payload.update(instructions="Follow the supplied annotation task. Return only the requested JSON object.",
                       input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                       max_output_tokens=max_tokens, store=False, reasoning={"effort":"low"})
        result = request(config, "/responses", payload)
        text = "".join(c.get("text", "") for item in result.get("output", [])
                       for c in item.get("content", []) if c.get("type") == "output_text")
    else:
        payload.update(messages=[{"role": "user", "content": prompt}], temperature=0,
                       max_tokens=max_tokens)
        if provider == "qwen":
            payload["enable_thinking"] = False
        if provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        result = request(config, "/chat/completions", payload)
        text = result["choices"][0]["message"].get("content", "")
    return {"requested_model": config["model"], "returned_model": result.get("model"),
            "generation_settings": ({"reasoning_effort":"low","max_output_tokens":max_tokens,"temperature":"provider_default"} if config.get("api")=="responses" else {"thinking":False,"temperature":0,"max_tokens":max_tokens}),
            "text": text, "usage": result.get("usage", {})}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=["gpt", "qwen", "deepseek"])
    args = parser.parse_args()
    try:
        print(json.dumps(complete(args.provider, 'Return only this JSON object: {"ok":true}', 256)))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
