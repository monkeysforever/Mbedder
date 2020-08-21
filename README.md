<div align="center">
<img src="https://user-images.githubusercontent.com/13309365/90921051-0885f480-e3af-11ea-9fbc-3e45e2bcc4b7.jpg" width="20%"/>
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

More advanced features include, outputting Sentence as well as Token level embeddings, the ability to apply task specific combination strategies to hidden states and token embeddings, adding custom pre-trained models with above mentioned architectures.




