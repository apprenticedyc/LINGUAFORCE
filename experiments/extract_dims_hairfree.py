# -*- coding: utf-8 -*-
"""Run the LINGUAFORCE parser through the local ccSwitch hairfree provider.

The API key is read transiently from ccSwitch's local SQLite database and is
never written to the repository or output JSONL files.
"""
import argparse
import json
import os
import pathlib
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_dims as ed


def load_hairfree_config():
    db = pathlib.Path.home() / ".cc-switch" / "cc-switch.db"
    con = sqlite3.connect(db)
    try:
        row = con.execute(
            "select settings_config from providers "
            "where id like 'hairfree-%' and app_type='codex' "
            "order by created_at desc limit 1"
        ).fetchone()
    finally:
        con.close()
    if not row:
        raise RuntimeError("No ccSwitch hairfree Codex provider found")
    config = json.loads(row[0])
    key = config.get("auth", {}).get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("ccSwitch hairfree provider has no API key")
    return key, "gpt-6-astra", "https://hairfree.corp.kuaishou.com/gateway/v1"


if __name__ == "__main__":
    # Reuse the established parser and output format, changing only auth/model.
    key, model, base_url = load_hairfree_config()
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["DEEPSEEK_MODEL"] = model

    # The original loader points to a Windows config. Replace it only in this
    # process so extract_dims can be used unchanged.
    ed.load_config = lambda: (key, model, base_url)
    ed.main()
