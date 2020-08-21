from Mbedder import (BertEmbedding, XLNetEmbedding, GPT2Embedding,
                     GPTEmbedding, TransfoXLEmbedding, XLMEmbedding,
                     RobertaEmbedding, DistilBertEmbedding, FlaubertEmbedding,
                     CamembertEmbedding, AlbertEmbedding, XLMRobertaEmbedding)

from transformers import (BertTokenizer, XLNetTokenizer, XLMTokenizer,
                          GPT2Tokenizer, OpenAIGPTTokenizer, TransfoXLTokenizer,
                          RobertaTokenizer, DistilBertTokenizer, FlaubertTokenizer,
                          CamembertTokenizer, AlbertTokenizer, XLMRobertaTokenizer)

import torch
import unittest


class TestEmbedding(unittest.TestCase):
    def forward(self) -> None:
        text = 'Beware of bugs in the above code; I have only proved it correct, not tried it.'
        output = self.tokenizer(text)
        input_ids = torch.tensor(output['input_ids']).unsqueeze(0)
        mask = torch.tensor(output['attention_mask']).unsqueeze(0)
        sentence_embedding, token_embedding = self.embedding(input=input_ids, mask=mask)
        embedding_size = self.embedding.embedding_size
        assert sentence_embedding.shape[1] == embedding_size
        assert token_embedding.shape[2] == 4 * embedding_size


class TestBertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(arch)
        self.embedding = BertEmbedding.from_pretrained(arch)


    def test_forward(self) -> None:
        self.forward()


class TestXLNetEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'xlnet-base-cased'
        self.tokenizer = XLNetTokenizer.from_pretrained(arch)
        self.embedding = XLNetEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestGPTEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'openai-gpt'
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(arch)
        self.embedding = GPTEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestTransfoXLEmbedding(unittest.TestCase):
    def test_forward(self) -> None:
        arch = 'transfo-xl-wt103'
        tokenizer = TransfoXLTokenizer.from_pretrained(arch)
        embedding = TransfoXLEmbedding.from_pretrained(arch)
        text = 'Beware of bugs in the above code; I have only proved it correct, not tried it.'
        output = tokenizer(text)
        input_ids = torch.tensor(output['input_ids']).unsqueeze(0)
        sentence_embedding, token_embedding = embedding(input=input_ids)
        embedding_size = embedding.embedding_size
        assert sentence_embedding.shape[1] == embedding_size
        assert token_embedding.shape[2] == 4 * embedding_size


class TestXLMEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'xlm-mlm-ende-1024'
        self.tokenizer = XLMTokenizer.from_pretrained(arch)
        self.embedding = XLMEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestRobertaEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(arch)
        self.embedding = RobertaEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestDistilBertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(arch)
        self.embedding = DistilBertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestCamembertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'camembert-base'
        self.tokenizer = CamembertTokenizer.from_pretrained(arch)
        self.embedding = CamembertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestAlbertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'albert-base-v1'
        self.tokenizer = AlbertTokenizer.from_pretrained(arch)
        self.embedding = AlbertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestFlaubertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'flaubert/flaubert_small_cased'
        self.tokenizer = FlaubertTokenizer.from_pretrained(arch)
        self.embedding = FlaubertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestXLMRobertaEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'xlm-roberta-base'
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(arch)
        self.embedding = XLMRobertaEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


def token_strategy(token_embeddings):
    embeds =[token_embeddings[:, :, -1, :],
            token_embeddings[:, :, -2, :],
            token_embeddings[:, :, -3, :],
            token_embeddings[:, :, -4, :]]
    best = embeds[0]
    for idx in range(1, len(embeds)):
        best = torch.max(best, embeds[idx])
    return best


def sentence_strategy(hidden_states):
    return torch.cat((hidden_states[-1], hidden_states[-2]), dim=2)


class TestTokenStrategy(unittest.TestCase):
    def setUp(self) -> None:
        arch = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(arch)
        self.embedding = BertEmbedding.from_pretrained(arch, token_strategy=token_strategy)

    def test_forward(self) -> None:
        text = 'Beware of bugs in the above code; I have only proved it correct, not tried it.'
        output = self.tokenizer(text)
        input_ids = torch.tensor(output['input_ids']).unsqueeze(0)
        mask = torch.tensor(output['attention_mask']).unsqueeze(0)
        sentence_embedding, token_embedding = self.embedding(input=input_ids, mask=mask)
        embedding_size = self.embedding.embedding_size
        assert token_embedding.shape[2] == embedding_size


class TestSentenceStrategy(unittest.TestCase):
    def setUp(self) -> None:
        arch = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(arch)
        self.embedding = BertEmbedding.from_pretrained(arch, sentence_strategy=sentence_strategy)

    def test_forward(self) -> None:
        text = 'Beware of bugs in the above code; I have only proved it correct, not tried it.'
        output = self.tokenizer(text)
        input_ids = torch.tensor(output['input_ids']).unsqueeze(0)
        mask = torch.tensor(output['attention_mask']).unsqueeze(0)
        sentence_embedding, token_embedding = self.embedding(input=input_ids, mask=mask)
        embedding_size = self.embedding.embedding_size
        assert sentence_embedding.shape[2] == 2 * embedding_size



