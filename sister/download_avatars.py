import pandas as pd
import requests
from pathlib import Path
from collections import defaultdict

df = pd.read_excel("/home/fsy23/UniPoly/sister/20260311.xlsx", header=None)
output_dir = Path("avatars")
output_dir.mkdir(exist_ok=True)

name_count = defaultdict(int)

for _, row in df.iterrows():
    name = str(row[0]).strip()
    url = str(row[1]).strip()
    if not name or not url or url == "nan":
        continue

    ext = Path(url.split("?")[0]).suffix or ".jpg"
    name_count[name] += 1
    suffix = str(name_count[name]) if name_count[name] > 1 else ""
    save_path = output_dir / f"{name}{suffix}{ext}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        save_path.write_bytes(resp.content)
        print(f"OK: {name} -> {save_path}")
    except Exception as e:
        print(f"FAIL: {name} ({url}): {e}")
