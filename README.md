## An Automatic Graph Construction Framework based on Large Language Models for Recommendation

### Introduction
This is the pytorch implementation of ***AutoGraph*** proposed in the paper [An Automatic Graph Construction Framework based on Large Language Models for Recommendation](https://arxiv.org/pdf/2412.18241).

### Required Packages
```
pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

pip install vllm transformers vector-quantize-pytorch
```

### Data preprocess
Download the original dataset, put them under directory 'Datasets', and run the notebooks in [Preprocess](./Preprocess).

### LLM Inference
Inference on user preference and item knowledge.
```bash
python gen_knowledge.py
```

### Encode LLM Knowledge
Encode the inferenced knowledges into vectors.
```bash
python gen_embedding.py
```

### Quantization
```bash
python quantization.py
```

### Enhance REC Model
```bash
python main.py --plus_gnn_embed
```

