# DMVF

## 项目简介

本项目实现论文 [From Single-Stream to Disentangled Factors: A Multi-View Framework for QoS Prediction](DMVF.pdf) 提出的 DMVF（Disentangled Multi-View Framework），用于稀疏观测下的时间感知 QoS（响应时间）预测。模型融合时间无关、时间相关和语义三个视角，通过分层注意力预测缺失的用户–服务–时间响应值。

## 环境搭建

以下以 CUDA 12.8 为例，其他版本可参考 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)。所有命令均在项目根目录执行。

```bash
conda create -n dmvf python=3.12 -y
conda activate dmvf

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas tqdm "transformers>=4.51,<5" accelerate safetensors huggingface_hub
```

## 数据预处理

将 Supplement WS-DREAM Dataset 2 的原始文件放到以下目录：

```text
dataset/Supplement_WSDREAM_dataset2/data/
├── rtdata.txt
├── userlist.txt
└── wslist.txt
```

使用 Hugging Face CLI 下载 [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) 到本地：

```bash
hf download Qwen/Qwen3-Embedding-4B --local-dir ./models/Qwen3-Embedding-4B
```

首次运行依次生成响应时间缓存、语义元数据和 Qwen 向量缓存：

```bash
python dataset/preprocess_rtdata.py
python dataset/preprocess_semantic.py
python dataset/build_semantic_cache.py --model-path ./models/Qwen3-Embedding-4B --device cuda:0
```

生成的 `rtdata_cache2.csv` 和 `semantic_qwen3_4b_cache.pt` 位于 `dataset/Supplement_WSDREAM_dataset2/`，`userlist_grid.txt` 和 `wslist_grid.txt` 位于其 `data/` 子目录。训练直接读取缓存，无需加载 Qwen。

预处理配置见 [preprocess_rtdata.py](dataset/preprocess_rtdata.py) 和 [preprocess_semantic.py](dataset/preprocess_semantic.py) 顶部参数区；Qwen 缓存的命令行参数见 [build_semantic_cache.py](dataset/build_semantic_cache.py) 的 `parse_args()`。已生成且配置一致的文件可直接复用。

## 运行

训练配置及命令行参数定义在 [config.py](config.py) 的 `Config` 和 `Config.parse_args()` 中。基础运行命令：

```bash
python train.py --device cuda:0 --train-ratio 0.1
```

训练与评估日志默认保存到 `log/`。
