# OpinioNet v2.0.0

OpinioNet_v2 是一个基于 BERT 的一阶段端到端实体关系抽取模型，专门用于观点挖掘（Opinion Mining）。它能够同时识别评价对象（Aspect）、观点词（Opinion）、分类（Category）以及情感极性（Polarity）。基于 [OpinioNet](https://github.com/eguilg/OpinioNet/tree/master).



## 环境配置

本项目建议使用 [uv](https://github.com/astral-sh/uv) 进行包管理。项目要求 Python 3.10。

### 1. 安装依赖

在项目根目录下执行以下命令创建虚拟环境并安装依赖：

```bash
# 创建虚拟环境并同步依赖
uv sync

# 或者使用 pip 安装
pip install -e .

# 然后进入环境
# Linux
source .venv/bin/activate    
# Windows
\.venv\Scripts\activate
```

### 2. 开发环境 (可选)

如果需要进行测试或代码格式化，请安装开发依赖：

```bash
uv sync --extra dev
```

## 数据准备

请确保数据存放在 `data/` 目录下，通常包括：
- `TRAIN/`: 训练集 CSV 文件。
- `TEST/`: 测试集 CSV 文件。

## 脚本使用指南

安装项目后，可以直接在终端调用以下命令：

### 1. MLM 预训练 (Pretraining)
使用无标签语料进行 Masked Language Modeling 预训练，以增强模型对特定领域文本的理解。

```bash
pretrain_model --config configs/pretrain.yaml
```

(本项目用不到，因为该比赛只提供了makeup的数据集)

### 2. 模型训练 (Training)
在标注数据集上进行有监督训练。

```bash
train_model --epochs 10
```


### 3. 模型评估 (Evaluation)
在测试集上评估模型性能，计算 F1 分数等指标。

```bash
# 如果不指定 model_path, 会默认选择 ckpt_dir 下的 xxx_best.pt 
eval_model --ckpt_dir models/roberta_large_makeup/20251226_160310 --model_path your_ckpt_name
```

### 4. 指标可视化 (Visualization)
可视化训练过程中的损失曲线和评估指标。

```bash
visualize_metrics --data_path models/roberta_makeup/20251219_183416/training_metrics.npy
```

### 5. （可选）模型融合评估

有融合的模型必须使用相同的 `Tokenizer`（即 `--base_model` 参数必须一致，比如都是 `roberta`），否则 Input IDs 会对不上。

```bash
ensemble_evaluate \
  --model_paths \
  "['models/roberta_makeup/.../roberta_best.pt', 'models/roberta_makeup/.../roberta_best.pt']"
```

注意：该脚本会同时将所有模型加载到显存中。如果显存不足（OOM），请减小 `--batch_size`（例如设为 1 或 2）

## 配置说明

本项目使用 `draccus` 处理配置。所有可执行文件的超参数（如 `learning_rate`, `batch_size`, `hidden_size` 等）均可以通过 YAML 配置文件或命令行参数进行修改。

示例：
```bash
train_model --learning_rate 2e-5 --batch_size 16
```

具体超参数可以通过：`--help` 查看

```bash
train_model  --help

eval_model --help

visualize_metrics --help
```

## 产生文件位置

* 默认情况下，**训练后**，该训练的配置会放置在 `models/{base_model}_{data_type}/{data_time}` 目录下，保存有本次训练的配置，最近的检查点，最优检查点，本次训练日志文件，以及训练和检验的metrics文件(`train_metrics.npy`)。

* **运行 `eval_model`** 并传递正确的模型路径后，程序会自动将预测结果放置在checkpoint同目录下的 `submit/Result.csv` 下。


* **运行 `ensemble_evaluate`** 并传递正确的模型路径后，程序会自动将预测结果放置在第一个checkpoint同目录下的 `submit_ensemble/Result.csv` 下。

* **运行 `visualize_metrics`** 并传递正确的模型路径后，程序会自动将可视化结果放置在checkpoint同目录下的 `training_metrics.png` 和 `validation_metrics.png` 下。