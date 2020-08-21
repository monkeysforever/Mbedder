<div align="center">
<img src="https://user-images.githubusercontent.com/13309365/90924089-6bc65580-e3b4-11ea-8f02-b2c1c024afb6.jpg" width="20%"/>
 <br />
 <br />
<a href="https://github.com/monkeysforever/Mbedder/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</div>

---
Mbedder is a language framework for adding contextual embeddings of pretrained language models to deep learning models.Mbedder is powered by <b>PyTorch</b> and <b>HuggingFace</b> and requires as less as 1 line of code to add embeddings and works similar to how the Embedding Layer works in PyTorch.

<details><summary>List of supported architectures</summary><p>
 
- **Bert**
- **XLNet**
- **Albert**
- **TransfoXL**
- **DistilBert**
- **Roberta**
- **XLM**
- **XLMRoberta**
- **GPT**
- **GPT2**
- **Flaubert**
</p></details>
The pretrained models for the mentioned architecures can be found <a href='https://huggingface.co/transformers/pretrained_models.html'>here.</a>

### Features
- Addition of embeddings with 1 line of code
- Embeddings can output Sentence as well as Token level embeddings
- Task specific combination strategies can be applied to hidden states and token embeddings
- Custom pre-trained hugging face transformer models can be used with Mbedder.

# Requirements and Installation
* [PyTorch](http://pytorch.org/) version >= 1.6.0
* Python version >= 3.6
* Transformer >= 3.0.2

Mbedder can be using Pip as follows
```
pip install mbedder
```

# Getting Started

A basic example of using a Mbedder Bert embedding is shown below:
```
import torch
from Mbedder import BertEmbedding

class BertClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.embedding = BertEmbedding.from_pretrained('bert-base-uncased')
        self.fc = torch.nn.Linear(self.embedding.embedding_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids, attention_mask, output_token_embeddings=False)
        logits = self.fc(x)
        return logits

```
More advanced examples can be found in the examples folder.

# License

Mbedder is MIT-licensed.






