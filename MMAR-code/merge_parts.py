import json
from pathlib import Path

OUT_DIR = "/data/shixun/MMAR/tmp_parts"
parts = ["part0.json", "part1.json", "part2.json", "part3.json"]

merged = []
for p in parts:
    path = Path(OUT_DIR) / p
    with open(path) as f:
        merged.extend(json.load(f))

# 保存合并后的 JSON
merged_path = Path(OUT_DIR) / "MMAR_merged.json"
with open(merged_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} samples into {merged_path}")
