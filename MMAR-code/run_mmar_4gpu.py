import json
import os
import argparse
from tqdm import tqdm
from swift.llm import InferRequest, PtEngine, RequestConfig
import re

# ========= CONFIG =========
MODEL_PATH = "/data/shixun/zhaohaozhe/model"
MMAR_JSON = "/data/shixun/MMAR/MMAR-meta.json"
AUDIO_ROOT = "/data/shixun/MMAR_full"
OUT_DIR = "/data/shixun/MMAR/tmp_parts"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== CRITICAL PROMPT ==========
SYSTEM_PROMPT = """You are an audio reasoning model.

You will be given:
- An audio clip
- A multiple-choice question with four options labeled A, B, C, D

You must decide which option is correct.

You must output ONLY ONE LETTER:
A, B, C, or D.

Do NOT output any words.
Do NOT explain.
Do NOT output the option text.
Only output A, B, C, or D.
"""

def build_prompt(question, choices):
    s = "Question:\n" + question + "\n\nOptions:\n"
    for i, c in enumerate(choices):
        s += f"{chr(ord('A')+i)}. {c}\n"
    s += "\nOutput only one letter: A, B, C, or D."
    return s

def build_message(audio_path, question, choices):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": build_prompt(question, choices)}
        ]}
    ]

# ========== Non-stream inference ==========
def run_one(engine, cfg, audio, q, choices):
    req = InferRequest(messages=build_message(audio, q, choices))
    result = engine.infer([req], cfg)

    first = list(result)[0]
    if isinstance(first, list):
        resp_obj = first[0]
    else:
        resp_obj = first

    resp = resp_obj.choices[0].message.content.strip().upper()

    # ========== Clean up for Audio-Reasoner ==========
    # 提取 <RESPONSE> 部分
    match = re.search(r"<RESPONSE>(.*?)</RESPONSE>", resp, re.DOTALL)
    if match:
        resp = match.group(1).strip()
    else:
        resp = resp.strip()

    # 只保留第一个合法字母
    for c in resp:
        if c in ["A", "B", "C", "D"]:
            return c

    return ""  # 如果没匹配到合法字母

# ========== Worker ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--part", type=int, required=True)   # 0,1,2,3
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}, part {args.part}")

    engine = PtEngine(
        MODEL_PATH,
        model_type="qwen2_audio",
        max_batch_size=1,
        trust_remote_code=True
    )

    cfg = RequestConfig(max_tokens=64, temperature=0, stream=False)

    with open(MMAR_JSON) as f:
        data = json.load(f)

    N = len(data)
    part_size = N // 4
    start = args.part * part_size
    end = N if args.part == 3 else (args.part + 1) * part_size

    out_path = f"{OUT_DIR}/part{args.part}.json"

    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for s in json.load(f):
                done[s["id"]] = s

    results = list(done.values())

    for i in tqdm(range(start, end)):
        s = data[i]
        if s["id"] in done:
            continue

        audio = os.path.join(AUDIO_ROOT, s["audio_path"].replace("./", ""))

        try:
            pred = run_one(engine, cfg, audio, s["question"], s["choices"])
        except Exception as e:
            print(f"Error at {i}:", e)
            pred = ""

        s["model_prediction"] = pred
        results.append(s)

        if len(results) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Finished part", args.part)

if __name__ == "__main__":
    main()
