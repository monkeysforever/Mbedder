from transformers import (BertModel, XLNetModel, GPT2Model,
                          OpenAIGPTModel, TransfoXLModel, XLMModel,
                          RobertaModel, DistilBertModel,
                          CamembertModel, AlbertModel,
                          XLMRobertaModel, FlaubertModel)

import torch

bert_archs = {'bert-base-uncased', 'bert-large-uncased',
              'bert-base-uncased', 'bert-large-uncased',
              'bert-base-multilingual-uncased', 'bert-base-multilingual-cased',
              'bert-base-chinese', 'bert-base-german-cased',
              'bert-large-uncased-whole-word-masking',
              'bert-large-cased-whole-word-masking',
              'bert-large-uncased-whole-word-masking-finetuned-squad',
              'bert-large-cased-whole-word-masking-finetuned-squad',
              'bert-base-cased-finetuned-mrpc', 'bert-base-german-dbmdz-cased',
              'bert-base-german-dbmdz-uncased', 'cl-tohoku/bert-base-japanese',
              'cl-tohoku/bert-base-japanese-whole-word-masking',
              'cl-tohoku/bert-base-japanese-char',
              'cl-tohoku/bert-base-japanese-char-whole-word-masking',
              'TurkuNLP/bert-base-finnish-cased-v1',
              'TurkuNLP/bert-base-finnish-uncased-v1',
              'wietsedv/bert-base-dutch-cased'}
xlnet_archs = {'xlnet-base-cased', 'xlnet-large-cased'}
gpt2_archs = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
xlm_archs = {'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024',
             'xlm-mlm-enro-1024', 'xlm-mlm-xnli15-1024',
             'xlm-mlm-tlm-xnli15-1024', 'xlm-clm-enfr-1024', 'xlm-clm-ende-1024',
             'xlm-mlm-17-1280', 'xlm-mlm-100-1280'}
roberta_archs = {'roberta-base', 'roberta-large', 'roberta-large-mnli',
                 'distilroberta-base', 'roberta-base-openai-detector',
                 'roberta-large-openai-detector'}

distilbert_archs = {'distilbert-base-uncased', 'distilbert-base-cased',
                    'distilbert-base-uncased-distilled-squad',
                    'distilbert-base-cased-distilled-squad', 'distilgpt2',
                    'distilbert-base-german-cased',
                    'distilbert-base-multilingual-cased'}

albert_archs = {'albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2',
                'albert-xlarge-v2', 'albert-xxlarge-v2'}

xlmroberta_archs = {'xlm-roberta-base', 'xlm-roberta-large'}

flaubert_archs = {'flaubert/flaubert_small_cased',
                  'flaubert/flaubert_base_cased',
                  'flaubert/flaubert_base_uncased',
                  'flaubert/flaubert_large_cased'}


class Embedding(torch.nn.Module):
    def __init__(self, arch, model, freeze, token_strategy, sentence_strategy):
        super(Embedding, self).__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = not freeze
        self.embedding_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.max_length = self.model.config.max_position_embeddings
        self.arch = arch
        self.token_strategy = token_strategy
        self.sentence_strategy = sentence_strategy

    def extra_repr(self) -> str:
        s = 'Architecture={arch}'
        s += ', Embedding Size={embedding_size}'
        s += ', Vocabulary Size={vocab_size}'
        s += ', Maximum Input Length={max_length}'
        return s.format(**self.__dict__)

    def get_embeddings(self, hidden_states, output_token_embeddings):
        if output_token_embeddings:
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = token_embeddings.permute(1, 2, 0, 3)
            if self.token_strategy is None:
                token_vectors = torch.cat((token_embeddings[:, :, -1, :],
                                           token_embeddings[:, :, -2, :],
                                           token_embeddings[:, :, -3, :],
                                           token_embeddings[:, :, -4, :]),
                                          dim=2)
            else:
                token_vectors = self.token_strategy(token_embeddings)

        if self.sentence_strategy is None:
            sentence_vector = torch.mean(hidden_states[-2], dim=1)
        else:
            sentence_vector = self.sentence_strategy(hidden_states)

        return (sentence_vector,) + (token_vectors,)


class BertEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in bert_archs, 'Invalid Architecture'
        assert type(model) == BertModel, 'Invalid Model'
        super(BertEmbedding, self).__init__(arch, model, freeze,
                                            token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='bert-base-uncased', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in bert_archs, 'Invalid Architecture'
        return cls(arch,
                   BertModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class XLNetEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in xlnet_archs, 'Invalid Architecture'
        assert type(model) == XLNetModel, 'Invalid Model'
        super(XLNetEmbedding, self).__init__(arch, model, freeze,
                                             token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='xlnet-base-cased', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in xlnet_archs, 'Invalid Architecture'
        return cls(arch,
                   XLNetModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[1]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class GPT2Embedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in gpt2_archs, 'Invalid Architecture'
        assert type(model) == GPT2Model, 'Invalid Model'
        super(GPT2Embedding, self).__init__(arch, model, freeze,
                                            token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='gpt2', freeze=True, token_strategy=None,
                        sentence_strategy=None):
        assert arch in gpt2_archs, 'Invalid Architecture'
        return cls(arch,
                   GPT2Model.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class GPTEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch == 'openai-gpt', 'Invalid Architecture'
        assert type(model) == OpenAIGPTModel, 'Invalid Model'
        super(GPTEmbedding, self).__init__(arch, model, freeze, token_strategy,
                                           sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='openai-gpt', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch == 'openai-gpt', 'Invalid Architecture'
        return cls(arch,
                   OpenAIGPTModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[1]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class TransfoXLEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch == 'transfo-xl-wt103', 'Invalid Architecture'
        assert type(model) == TransfoXLModel, 'Invalid Model'
        super(TransfoXLEmbedding, self).__init__(arch, model, freeze,
                                                 token_strategy,
                                                 sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='gpt2', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch == 'transfo-xl-wt103', 'Invalid Architecture'
        return cls(arch,
                   TransfoXLModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, output_token_embeddings=True):
        outputs = self.model(input_ids=input, output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class XLMEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in xlm_archs, 'Invalid Architecture'
        assert type(model) == XLMModel, 'Invalid Model'
        super(XLMEmbedding, self).__init__(arch, model, freeze,
                                           token_strategy,
                                           sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='xlm-mlm-en-2048', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in xlm_archs, 'Invalid Architecture'
        return cls(arch,
                   XLMModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[1]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class RobertaEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in roberta_archs, 'Invalid Architecture'
        assert type(model) == RobertaModel, 'Invalid Model'
        super(RobertaEmbedding, self).__init__(arch, model, freeze,
                                               token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='bert-base-uncased', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in roberta_archs, 'Invalid Architecture'
        return cls(arch,
                   RobertaModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class DistilBertEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in distilbert_archs, 'Invalid Architecture'
        assert type(model) == DistilBertModel, 'Invalid Model'
        super(DistilBertEmbedding, self).__init__(arch, model, freeze,
                                                  token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='distilbert-base-uncased', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in distilbert_archs, 'Invalid Architecture'
        return cls(arch,
                   DistilBertModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[1]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class CamembertEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch == 'camembert-base', 'Invalid Architecture'
        assert type(model) == CamembertModel, 'Invalid Model'
        super(CamembertEmbedding, self).__init__(arch, model, freeze,
                                                 token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='camembert-base', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch == 'camembert-base', 'Invalid Architecture'
        return cls(arch,
                   CamembertModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class AlbertEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in albert_archs, 'Invalid Architecture'
        assert type(model) == AlbertModel, 'Invalid Model'
        super(AlbertEmbedding, self).__init__(arch, model, freeze,
                                              token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='albert-base-v1', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in albert_archs, 'Invalid Architecture'
        return cls(arch,
                   AlbertModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class XLMRobertaEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in xlmroberta_archs, 'Invalid Architecture'
        assert type(model) == XLMRobertaModel, 'Invalid Model'
        super(XLMRobertaEmbedding, self).__init__(arch, model, freeze,
                                                  token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='xlm-roberta-base', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in xlmroberta_archs, 'Invalid Architecture'
        return cls(arch,
                   XLMRobertaModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[2]
        return self.get_embeddings(hidden_states, output_token_embeddings)


class FlaubertEmbedding(Embedding):
    def __init__(self, arch, model, freeze=True, token_strategy=None,
                 sentence_strategy=None) -> None:
        assert arch in flaubert_archs, 'Invalid Architecture'
        assert type(model) == FlaubertModel, 'Invalid Model'
        super(FlaubertEmbedding, self).__init__(arch, model, freeze,
                                                token_strategy, sentence_strategy)

    @classmethod
    def from_pretrained(cls, arch='flaubert/flaubert_small_cased', freeze=True,
                        token_strategy=None, sentence_strategy=None):
        assert arch in flaubert_archs, 'Invalid Architecture'
        return cls(arch,
                   FlaubertModel.from_pretrained(arch),
                   freeze,
                   token_strategy,
                   sentence_strategy)

    def forward(self, input, mask=None, output_token_embeddings=True):
        outputs = self.model(input_ids=input, attention_mask=mask,
                             output_hidden_states=True)
        hidden_states = outputs[1]
        return self.get_embeddings(hidden_states, output_token_embeddings)
