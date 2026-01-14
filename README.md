数据与模型下载
由于 Audio-Reasoner 的权重和配置文件以及 MMAR 数据集文件较大，建议从以下地址下载：

Audio-Reasoner 权重和配置文件：https://huggingface.co/zhifeixie/Audio-Reasoner/tree/main

MMAR 数据集：https://huggingface.co/datasets/BoJack/MMAR

下载完成后，请在以下脚本中替换相应的路径：
inference.py：用于测试单个音频文件。
run_mmar_4gpu.py：用于测试整个 MMAR 数据集，需要 4 个 GPU 支持。
