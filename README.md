1.数据与模型下载

由于 Audio-Reasoner 的权重和配置文件以及 MMAR 数据集文件过于庞大（几十个G），建议从以下地址下载：

Audio-Reasoner 权重和配置文件：https://huggingface.co/zhifeixie/Audio-Reasoner/tree/main

MMAR 数据集：https://huggingface.co/datasets/BoJack/MMAR

下载完成后，请在以下脚本中替换相应的路径：

inference.py：用于测试单个音频文件。

run_mmar_4gpu.py：用于测试整个 MMAR 数据集，需要 4 个 GPU 支持。可以根据服务器的实际 GPU 数量进行调整

2.代码文件夹分类

（1）MMAR-code文件夹包含：

evaluation.py：评估模型跑完测试集后得到的结果，会显示模型做各种题目的准确率

run_mmar_4gpu.py：调用模型，在整个MMAR数据集集进行测试，同时由于我服务器的缘故，该代码同时将MMAR中的1000条数据平均分成4份，交给4个GPU并行计算，以提高效率

merge_parts.py：合并run_mmar_4gpu.py生成的四个.json文件，生成的最终结果给evaluation.py，供其评判


（2）Audio-Reasoner文件夹包含：

inference.py：用于运行模型推理的脚本，负责加载模型Audio-Reasoner来对单一数据进行处理

requirements.txt：项目依赖清单，这是Python项目的标准依赖管理文件，列出了运行项目所需的所有第三方库及其版本。

3.项目源GitHub地址

关于Audio-Reasoner的详细配置步骤可以参考：https://github.com/xzf-thu/Audio-Reasoner

关于MMAR数据集的详细信息可以参考：https://github.com/ddlBoJack/MMAR
