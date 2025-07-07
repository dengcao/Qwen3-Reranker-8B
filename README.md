# 2025年7月8日 更新说明

使用了Vllm最新的包（[v0.9.2rc2](https://github.com/vllm-project/vllm/releases/tag/v0.9.2rc2)）更新了Docker镜像：[dengcao/vllm-openai: v0.9.2rc2](https://hub.docker.com/r/dengcao/vllm-openai/tags)。

**更新方法：** 1、重新下载docker-compose.yaml文件覆盖旧文件。2、删除对应容器后，cd切换到项目根目录，重新执行：docker compose up -d



# 2025年6月26日 更新说明

 ·本项目旨在解决Qwen3-Reranker-8B模型无法通过Vllm平台直接部署的问题。

 ·采用vllm最新的开发版制作了Docker镜像dengcao/vllm-openai : v0.9.2-dev，经测试正常，可放心使用。
 
 ·修复了Qwen3-Reranker-8B排序结果可能不准确的问题。
 
 ### 注意：2025年6月26日之前已下载本项目的，请删除对应的docker容器和文件后重新使用此方法部署，即可完美在Vllm上运行Qwen3-Reranker-8B模型。


# 2025年6月20日 更新说明

自从Qwen3-Reranker系列模型发布以来，迅速在向量模型和重排模型中掀起了使用热潮，但遗憾的是，无法正常使用Vllm部署Qwen3-Reranker-8B模型，截止目前，Vllm官方也没有更新补丁支持，预计官方最快在Vllm v0.9.2中才支持Qwen3-Reranker。作为过渡，于是做了这个版本供大家暂时使用。


## Docker desktop（Windows用户）使用方法如下：

（前提是先在windows部署好Docker desktop）

1、下载本项目到windows任意目录。比如：C:\Users\Administrator\vLLM。

2、下载Qwen3-Reranker-8B模型放到目录：models\Qwen3-Reranker-8B。（不懂的可以从魔搭下载：https://www.modelscope.cn/models/dengcao/Qwen3-Reranker-8B）

3、打开：Windows PowerShell，输入：wsl，确定后进入WSL。依次执行下列命令：

cd /mnt/c/Users/Administrator/vLLM

docker compose up -d

等待下载docker镜像和加载容器完成即可。

### 当然以上命令也可以直接在windows命令窗口这样执行：

cd C:\Users\Administrator\vLLM

docker compose up -d

## Linux用户的Docker版本，可参考以上方法。

## 调用Qwen3-Reranker-8B模型API接口：

### Docker内的容器APP调用：

API请求地址：http://host.docker.internal:8012/v1/rerank

请求Key：NOT_NEED

### Docker外部的APP调用：

API请求地址：http://localhost:8012/v1/rerank

请求Key：NOT_NEED

### 此方法已经在FastGPT上测试通过，可正常排序。

此项目的解决方法和灵感来自：https://github.com/vllm-project/vllm/pull/19260 ，特感谢相关开发者。



---
license: apache-2.0
base_model:
- Qwen/Qwen3-8B-Base
library_name: transformers
pipeline_tag: text-ranking
---
# Qwen3-Reranker-8B

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="400"/>
<p>

## Highlights

The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

**Exceptional Versatility**: The embedding model has achieved state-of-the-art performance across a wide range of downstream application evaluations. The 8B size embedding model ranks No.1 in the MTEB multilingual leaderboard (as of June 5, 2025, score 70.58), while the reranking model excels in various text retrieval scenarios.

**Comprehensive Flexibility**: The Qwen3 Embedding series offers a full spectrum of sizes (from 0.6B to 8B) for both embedding and reranking models, catering to diverse use cases that prioritize efficiency and effectiveness. Developers can seamlessly combine these two modules. Additionally, the embedding model allows for flexible vector definitions across all dimensions, and both embedding and reranking models support user-defined instructions to enhance performance for specific tasks, languages, or scenarios.

**Multilingual Capability**: The Qwen3 Embedding series offer support for over 100 languages, thanks to the multilingual capabilites of Qwen3 models. This includes various programming languages, and provides robust multilingual, cross-lingual, and code retrieval capabilities.


## Model Overview

**Qwen3-Reranker-8B** has the following features:

- Model Type: Text Reranking
- Supported Languages: 100+ Languages
- Number of Paramaters: 8B
- Context Length: 32k

For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to our [blog](https://qwenlm.github.io/blog/qwen3-embedding/), [GitHub](https://github.com/QwenLM/Qwen3-Embedding).

## Qwen3 Embedding Series Model list

| Model Type       | Models               | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------------|----------------------|------|--------|-----------------|---------------------|-------------|----------------|
| Text Embedding   | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | 28     | 32K             | 1024                | Yes         | Yes            |
| Text Embedding   | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)   | 4B   | 36     | 32K             | 2560                | Yes         | Yes            |
| Text Embedding   | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)   | 8B   | 36     | 32K             | 4096                | Yes         | Yes            |
| Text Reranking   | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B | 28     | 32K             | -                   | -           | Yes            |
| Text Reranking   | [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)   | 4B   | 36     | 32K             | -                   | -           | Yes            |
| Text Reranking   | [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B)   | 8B   | 36     | 32K             | -                   | -           | Yes            |

> **Note**:
> - `MRL Support` indicates whether the embedding model supports custom dimensions for the final embedding. 
> - `Instruction Aware` notes whether the embedding or reranking model supports customizing the input instruction according to different tasks.
> - Our evaluation indicates that, for most downstream tasks, using instructions (instruct) typically yields an improvement of 1% to 5% compared to not using them. Therefore, we recommend that developers create tailored instructions specific to their tasks and scenarios. In multilingual contexts, we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English.


## Usage

With Transformers versions earlier than 4.51.0, you may encounter the following error:
```
KeyError: 'qwen3'
```

### Transformers Usage

```python
# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-8B", padding_side='left')

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-8B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-8B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = ["What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
```

📌 **Tip**: We recommend that developers customize the `instruct` according to their specific scenarios, tasks, and languages. Our tests have shown that in most retrieval scenarios, not using an `instruct` on the query side can lead to a drop in retrieval performance by approximately 1% to 5%.

## Evaluation

| Model                              | Param  | MTEB-R  | CMTEB-R | MMTEB-R | MLDR   | MTEB-Code | FollowIR |
|------------------------------------|--------|---------|---------|---------|--------|-----------|----------|
| **Qwen3-Embedding-0.6B**               | 0.6B   | 61.82   | 71.02   | 64.64   | 50.26  | 75.41     | 5.09     |
| Jina-multilingual-reranker-v2-base | 0.3B   | 58.22   | 63.37   | 63.73   | 39.66  | 58.98     | -0.68    |
| gte-multilingual-reranker-base                      | 0.3B   | 59.51   | 74.08   | 59.44   | 66.33  | 54.18     | -1.64    |
| BGE-reranker-v2-m3                 | 0.6B   | 57.03   | 72.16   | 58.36   | 59.51  | 41.38     | -0.01    |
| **Qwen3-Reranker-0.6B**                | 0.6B   | 65.80   | 71.31   | 66.36   | 67.28  | 73.42     | 5.41     |
| **Qwen3-Reranker-4B**                  | 4B   | **69.76** | 75.94   | 72.74   | 69.97  | 81.20     | **14.84** |
| **Qwen3-Reranker-8B**                  | 8B     | 69.02   | **77.45** | **72.94** | **70.19** | **81.22** | 8.05     |

> **Note**:  
> - Evaluation results for reranking models. We use the retrieval subsets of MTEB(eng, v2), MTEB(cmn, v1), MMTEB and MTEB (Code), which are MTEB-R, CMTEB-R, MMTEB-R and MTEB-Code.
> - All scores are our runs based on the top-100 candidates retrieved by dense embedding model [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## Citation
If you find our work helpful, feel free to give us a cite.

```
@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}
```
