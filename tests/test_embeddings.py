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
        tokenizer = XLNetTokenizer.from_pretrained(arch)
        embedding = XLNetEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestGPTEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'openai-gpt'
        tokenizer = OpenAIGPTTokenizer.from_pretrained(arch)
        embedding = GPTEmbedding.from_pretrained(arch)

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
        tokenizer = XLMTokenizer.from_pretrained(arch)
        embedding = XLMEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class RobertaEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(arch)
        embedding = RobertaEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestDistilBertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(arch)
        embedding = DistilBertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestCamembertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'camembert-base'
        tokenizer = CamembertTokenizer.from_pretrained(arch)
        embedding = CamembertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestAlbertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'xlm-mlm-ende-1024'
        tokenizer = AlbertTokenizer.from_pretrained(arch)
        embedding = AlbertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestFlaubertEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'xlm-mlm-ende-1024'
        tokenizer = FlaubertTokenizer.from_pretrained(arch)
        embedding = FlaubertEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


class TestXLMRobertaEmbedding(TestEmbedding):
    def setUp(self) -> None:
        arch = 'xlm-mlm-ende-1024'
        tokenizer = XLMRobertaTokenizer.from_pretrained(arch)
        embedding = TestXLMRobertaEmbedding.from_pretrained(arch)

    def test_forward(self) -> None:
        self.forward()


def token_strategy(token_embeddings):
    return torch.max((token_embeddings[:, :, -1, :],
                      token_embeddings[:, :, -2, :],
                      token_embeddings[:, :, -3, :],
                      token_embeddings[:, :, -4, :]),
                     dim=2)


def sentence_strategy(hidden_states):
    return torch.concat((hidden_states[-1], hidden_states[-2]), dim=1)


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
        assert sentence_embedding.shape[1] == 2 * embedding_size



