import os
from typing import List
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
from swift.plugin import InferStats

# ============================
# 1. Audio-Reasoner Prompt
# ============================

system = """You are an audio deep-thinking model.
Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>.
The <THINK> section should be further divided into four parts:
<PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.
"""

def get_message(audiopath, prompt):
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audiopath},
                {"type": "text", "text": prompt}
            ]
        }
    ]

# ============================
# 2. Stream inference
# ============================

def infer_stream(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(
        max_tokens=2048,
        temperature=0,
        stream=True
    )
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])

    query = infer_request.messages[1]["content"][1]["text"]
    print(f"\nQuery: {query}\nResponse:\n")

    output = ""
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        delta = resp_list[0].choices[0].delta.content
        print(delta, end="", flush=True)
        output += delta

    print("\n\nMetric:", metric.compute())
    return output


# ============================
# 3. Load Audio-Reasoner model
# ============================

# 🔥 你真正的模型路径（你已经给过）
MODEL_PATH = "/data/shixun/zhaohaozhe/model"

engine = PtEngine(
    MODEL_PATH,
    model_type="qwen2_audio",   # 来自 config.json
    max_batch_size=1,           # Audio-Reasoner 非常吃显存，必须 1
    trust_remote_code=True      # Qwen2Audio 是自定义架构
)

# ============================
# 4. Audio-Reasoner API
# ============================

def audioreasoner_gen(audiopath, prompt):
    messages = get_message(audiopath, prompt)
    request = InferRequest(messages=messages)
    return infer_stream(engine, request)


# ============================
# 5. Main
# ============================

def main():
    # 你的测试音频
    audiopath = "assets/test.wav"

    # Audio-Reasoner 风格问题
    prompt = "Which of the following best describes the rhythmic feel and time signature of the song?"

    audioreasoner_gen(audiopath, prompt)


if __name__ == "__main__":
    main()
