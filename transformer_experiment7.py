'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
let's start transformer!
the code refered to
https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

-dataset from Multi30k
-my experiment
-trainable word embedding, positional embedding with relative position
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
imports
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import spacy # for tokenizer

from glove import Glove

import numpy as np
import sys
import os
import os.path
import math
import dill as pickle
import gzip
import argparse 
import copy

import hyperparameters2 as hparams
import customutils as utils

def program_loop():
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    preparing data and environment
    
    # torchtext==0.6.0
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''sample for time saving'''
    sample_valid_size = 300
    sample_test_size = 50

    model_name = 'transformer_en_de_exp_1'
    model_filepath = f'{os.getcwd()}/{model_name}.pt'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')


    # accumulate_loss = 0
    logger = TensorBoardLogger('runs', name='transformer_exp7', default_hp_metric=False)

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    '''load data'''
    # tokenize = tokenize_en,
    SRC = Field(
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)
    # tokenize = tokenize_de,
    TRG = Field(
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    st = utils.time.time()
    train, valid, test = Multi30k.splits(exts=('.en', '.de'),
                                         fields=(SRC, TRG))
    et = utils.time.time()
    m, s = utils.crop_time(st, et)
    print(f'data split completed | time : {m}m {s}s')
    sys.stdout.flush()

    st = utils.time.time()
    SRC.build_vocab(train, valid, test)
    et = utils.time.time()
    m, s = utils.crop_time(st, et)
    print(f"SRC build success | time : {m}m {s}s")
    sys.stdout.flush()

    st = utils.time.time()
    TRG.build_vocab(train, valid, test)
    et = utils.time.time()
    m, s = utils.crop_time(st, et)
    print(f"TRG build success | time : {m}m {s}s")
    sys.stdout.flush()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    BucketIterator
    
    if your computer has just a few cores, maybe able to go with it.
    since bucketiterator can not fix number of workers,
    DataLoader is recommended
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # st = utils.time.time()
    # train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    #     (train, valid, test),
    #     batch_size=hparams.batch_size,
    #     device=device)
    # et = utils.time.time()
    # m, s = utils.crop_time(st, et)
    # print(f"bucketiterator splits complete | time : {m}m {s}s")
    # sys.stdout.flush(pl_logs)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    since BucketIterator is too slow (low GPU, high CPU),
    encouraged to use torch.utils.data.DataLoader
    by https://github.com/pytorch/text/issues/664
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    st = utils.time.time()

    train_ds = sorted([(len(each.src),
                        len(each.trg),
                        [SRC.vocab[i] for i in each.src],
                        [TRG.vocab[i] for i in each.trg])
                        for i, each in enumerate(train)], key=lambda data:(data[0], data[1]), reverse=True)

    valid_ds = sorted([(len(each.src),
                        len(each.trg),
                        [SRC.vocab[i] for i in each.src],
                        [TRG.vocab[i] for i in each.trg])
                        for i, each in enumerate(valid)], key=lambda data:(data[0], data[1]), reverse=True)

    test_ds = sorted([(len(each.src),
                       len(each.trg),
                       [SRC.vocab[i] for i in each.src],
                       [TRG.vocab[i] for i in each.trg])
                      for i, each in enumerate(test)], key=lambda data:(data[0], data[1]), reverse=True)

    # when max_len_sentence is needed before pad_data

    max_len_sentence = (0,0)
    max_len_sentence = max(*[(i,len(vars(train.examples[i])['src'])) for i in range(len(train.examples))], key=lambda x:x[1])
    max_len_sentence = max(max_len_sentence, *[(i,len(vars(train.examples[i])['trg'])) for i in range(len(train.examples))], key=lambda x:x[1])
    max_len_sentence = max(max_len_sentence, *[(i,len(vars(valid.examples[i])['src'])) for i in range(len(valid.examples))], key=lambda x:x[1])
    max_len_sentence = max(max_len_sentence, *[(i,len(vars(valid.examples[i])['trg'])) for i in range(len(valid.examples))], key=lambda x:x[1])
    max_len_sentence = max(max_len_sentence, *[(i,len(vars(test.examples[i])['src'])) for i in range(len(test.examples))], key=lambda x:x[1])
    max_len_sentence = max(max_len_sentence, *[(i,len(vars(test.examples[i])['trg'])) for i in range(len(test.examples))], key=lambda x:x[1])

    print('max_len_sentence: ', max_len_sentence[1], vars(train.examples[max_len_sentence[0]])['src'], '/', vars(train.examples[max_len_sentence[0]])['trg'], 'length', max_len_sentence[1])
    sys.stdout.flush()

    max_len_sentence = max_len_sentence[1]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    collate function for DataLoader
    
    if you want to pad data in a batch, you can add collate_fn in DataLoader
    or you can pad for all sorted data by length before you set DataLoader
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def pad_data(data):
        '''Find max length of the mini-batch'''

        '''look data as column'''
        global max_len_sentence

        max_len_trg = max(list(zip(*data))[1])
        max_len_src = max(list(zip(*data))[0])
        src_ = list(zip(*data))[2]
        trg_ = list(zip(*data))[3]

        '''eos + pad'''
        padded_src = torch.stack([torch.cat((torch.as_tensor(txt).requires_grad_(False), torch.as_tensor([SRC.vocab.stoi[SRC.eos_token]]+([SRC.vocab.stoi[SRC.pad_token]] * (max_len_src - len(txt)))).requires_grad_(False).long())) for txt in src_])
        '''init token'''
        padded_src = torch.cat((torch.as_tensor([[SRC.vocab.stoi[SRC.init_token]]] * len(data)).requires_grad_(False), padded_src), dim=1)

        '''eos + pad'''
        padded_trg = torch.stack([torch.cat((torch.as_tensor(txt).requires_grad_(False), torch.as_tensor([TRG.vocab.stoi[TRG.eos_token]]+([TRG.vocab.stoi[TRG.pad_token]] * (max_len_trg - len(txt)))).requires_grad_(False).long())) for txt in trg_])
        '''init token'''
        padded_trg = torch.cat((torch.as_tensor([[TRG.vocab.stoi[TRG.init_token]]] * len(data)).requires_grad_(False), padded_trg), dim=1)
        # max_len_sentence = max(max_len_sentence, len(padded_src[0]), len(padded_trg[0]))
        '''for pad all before declaring DataLoader'''
        return [(s,t) for s,t in zip(padded_src, padded_trg)]
        '''for collate_fn parameter when declaring DataLoader'''
        # return padded_src, padded_trg

    train_ds = pad_data(train_ds)
    valid_ds = pad_data(valid_ds)
    test_ds = pad_data(test_ds)

    et = utils.time.time()
    m, s = utils.crop_time(st, et)

    print(f"data is ready")
    print(f"train_data : {len(train.examples)}")
    print(f"valid_data : {len(valid.examples)}")
    print(f"test_data : {len(test.examples)}")
    print(f"data example : {vars(train.examples[0])['src']}, {vars(train.examples[0])['trg']}")
    print(f"time : {m}m {s}s")
    sys.stdout.flush()

    train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)

    # example
    # for i, batch in enumerate(dataloader):
    #     src = batch[0]
    #     trg = batch[1]
    #     break

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    embedding
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class PretrainedEmbedding(pl.LightningModule):
        def __init__(self, stoi_vocab, d_model, filename, filename2=None):
            super().__init__()
            # global pretrained_embed

            # if pretrained_embed is not None:
            #     self.src_embed_mtrx = pretrained_embed.src_embed_mtrx
            #     self.trg_embed_mtrx = pretrained_embed.trg_embed_mtrx
            #     return

            # src_vocabsize = len(SRC.vocab.stoi)
            # trg_vocabsize = len(TRG.vocab.stoi)
            vocab_size = len(stoi_vocab)

            # self.register_buffer('src_embed_mtrx', torch.randn(src_vocabsize, hparams.d_model))
            # self.register_buffer('trg_embed_mtrx', torch.randn(src_vocabsize, hparams.d_model))
            self.embed_mtrx = torch.randn(vocab_size, d_model)

            '''
            for word2vec
            '''
            # src_word2vec = Word2Vec.load('src_embedd.model')
            # trg_word2vec = Word2Vec.load('trg_embedd.model')

            # for i in range(src_vocabsize):
            #     word = list(SRC.vocab.stoi.keys())[i]
            #     if word in src_word2vec.wv.index2word:
            #         src_embed_mtrx[SRC.vocab.stoi[word]] = torch.tensor(src_word2vec.wv[word].copy()).to(device)
            #
            # for i in range(trg_vocabsize):
            #     word = list(TRG.vocab.stoi.keys())[i]
            #     if word in trg_word2vec.wv.index2word:
            #         trg_embed_mtrx[TRG.vocab.stoi[word]] = torch.tensor(trg_word2vec.wv[word].copy()).to(device)

            '''
            for glove
            '''
            glove = Glove()
            glove_ = glove.load(filename)
            # src_glove = glove.load('src_glove.model')
            # trg_glove = glove.load('trg_glove.model')

            for word in list(stoi_vocab.keys()):
                if word in glove_.dictionary:
                    self.embed_mtrx[stoi_vocab[word]] = torch.tensor(glove_.word_vectors[glove_.dictionary[word]].copy())
            if filename2:
                glove_ = glove.load(filename)
                # src_glove = glove.load('src_glove.model')
                # trg_glove = glove.load('trg_glove.model')

                for word in list(stoi_vocab.keys()):
                    if word in glove_.dictionary:
                        self.embed_mtrx[stoi_vocab[word]] = torch.tensor(glove_.word_vectors[glove_.dictionary[word]].copy())


            # for word in list(TRG.vocab.stoi.keys()):
            #     if word in trg_glove.dictionary and self.trg_dim:
            #         self.trg_embed_mtrx[SRC.vocab.stoi[word]] = torch.tensor(trg_glove.word_vectors[trg_glove.dictionary[word]].copy())

            self.embed = nn.Embedding(vocab_size, d_model).from_pretrained(self.embed_mtrx).requires_grad_(False)

            print("pretrained word embeddings loaded")
            sys.stdout.flush()

        def forward(self, src):
            return self.embed(src)
        def training_step(self, src):
            return self.forward(src)


    # pretrained_embedding = PretrainedEmbedding()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    positional encoding
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def get_sinusoid_encoding_table(t_seq, d_model):
        def cal_angle(pos, i_model):
            return pos / np.power(10000, 2 * (i_model // 2) / d_model)

        def get_position_vec(pos):
            return [cal_angle(pos, i_model) for i_model in range(d_model)]

        sinusoid_table = np.array([get_position_vec(pos) for pos in range(t_seq)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return sinusoid_table

    # sinusoid_encoding_table = torch.FloatTensor(get_sinusoid_encoding_table(max_len_sentence, hparams.d_model)).to(device)
    class PositionalEncoding(pl.LightningModule):
        def __init__(self, t_seq, d_model):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True
            # self.sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(t_seq, d_model))
            self.register_buffer("sinusoid_table", torch.FloatTensor(get_sinusoid_encoding_table(t_seq, d_model)))

        def forward(self, x):
            x_len = x.shape[1]
            # sinusoid_table = self.sinusoid_table.type_as(x)
            with torch.no_grad():
                # with profiler.record_function("POSITIONAL ENCODING"):
                result = torch.add(x, self.sinusoid_table[:x_len, :])
            # del sinusoid_table
            return result

        def training_step(self, batch, batch_idx):
            x = batch
            return self.forward(x)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Self Attention
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class MultiHeadAttentionLayer(pl.LightningModule):
        def __init__(self, d_k, d_v, d_model, n_heads, dropout_ratio, max_length=100, self_=True):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True
            # since d_v * n_heads = d_model in the paper,
            assert d_model % n_heads == 0

            self.d_k = d_k
            self.d_v = d_v
            self.d_model = d_model
            self.n_heads = n_heads
            self.max_length = max_length
            self.self_ = self_

            self.w_q = nn.Linear(d_model, d_k * n_heads)
            nn.init.xavier_uniform_(self.w_q.weight.data)
            self.w_k = nn.Linear(d_model, d_k * n_heads)
            nn.init.xavier_uniform_(self.w_k.weight.data)
            self.w_v = nn.Linear(d_model, d_v * n_heads)
            nn.init.xavier_uniform_(self.w_v.weight.data)
            self.w_o = nn.Linear(d_v * n_heads, d_model)
            nn.init.xavier_uniform_(self.w_o.weight.data)

            self.dropout = nn.Dropout(dropout_ratio)
            scale = torch.sqrt(torch.FloatTensor([self.d_k]))
            self.register_buffer("scale", scale)

            if self.self_:
                self.pos_attention = PositionAttentionLayer(d_p=hparams.d_p,
                                                            d_v=self.d_v,
                                                            dropout_ratio=hparams.dropout_ratio,
                                                            max_length=self.max_length)

        def forward(self, query, key, value, mask=None):
            # with profiler.record_function("MultiHeadAttentionLayer"):
            batch_size = query.shape[0]
            query_len = query.shape[1]
            key_len = key.shape[1]
            value_len = value.shape[1]
            Q = self.w_q(query)
            K = self.w_k(key)
            V = self.w_v(value)

            # make seperate heads
            Q = Q.view(batch_size, -1, self.n_heads, self.d_k).permute(0,2,1,3)
            K = K.view(batch_size, -1, self.n_heads, self.d_k).permute(0,2,1,3)
            V = V.view(batch_size, -1, self.n_heads, self.d_k).permute(0,2,1,3)

            # Q : [batch_size, n_heads, query_len, head_dim(d_k)]

            self.scale = self.scale.type_as(query)
            similarity = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
            # similarity: [batch_size, n_heads, query_len, key_len]

            if mask is not None:
                similarity = similarity.masked_fill(mask==0, -1e10)

            similarity_norm = torch.softmax(similarity, dim=-1)
            # similarity_norm : [batch_size, n_heads, query_len, key_len]

            # dot product attention
            # value_relative = torch.sum(value_relative, dim=1).to(self.device)
            x = torch.matmul(self.dropout(similarity_norm), V)

            # x: [batch_size, n_heads, query_len, value_len(d_v)]
            if self.self_:
                x += self.pos_attention(query_len)
            x = x.permute(0, 2, 1, 3).contiguous()
            # x: [batch_size, query_len, n_heads, d_v]
            x = x.view(batch_size, -1, self.d_model)
            # x: [batch_size, query_len, d_model]
            x = self.w_o(x)
            # x: [batch_size, query_len, d_model]
            return x, similarity_norm

        def training_step(self, batch, batch_idx):
            try:
                x, y, z = batch
                return self.forward(x,y,z)
            except:
                x, y, z, m = batch
                return self.forward(x, y, z, m)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer] #, [lr_scheduler]

    class PositionAttentionLayer(pl.LightningModule):
        def __init__(self, d_p, d_v, dropout_ratio, max_length=100, k=hparams.k):
            super().__init__()

            @property
            def automatic_optimization(self):
                return True

            self.d_p = d_p
            self.d_v = d_v
            self.max_length = max_length
            self.overall_embedding = nn.Embedding(max_length, self.d_p)
            self.local_embedding = nn.Embedding(2*k+1, self.d_p)
            self.abs_pos_embedding = nn.Embedding(max_length, self.d_v)

            self.local_weight = nn.Embedding(max_length, 1)

            relation_map = torch.IntTensor(self.max_length, self.max_length)
            for i in range(max_length):
                relation_map[i] = torch.arange(start=k - i, end=k - i + max_length)
            relation_map = torch.clip(relation_map, min=0, max=2 * k)
            self.register_buffer("relation_map", relation_map)

            self.register_buffer("absolute_map", torch.arange(max_length))

        def forward(self, query_len):
            local_map = self.relation_map[:query_len, :query_len]
            overall_map = self.absolute_map[:query_len]

            local = self.local_embedding(local_map)
            overall = self.overall_embedding(overall_map)
            abs_pos = self.abs_pos_embedding(overall_map)
            local_weight = self.local_weight(overall_map)

            # local : [query_len, query_len, d_p]
            # overall : [query_len, d_p]
            # abs_pos : [query_len, d_v]

            overall = overall.view(query_len, self.d_p).permute(1,0)
            similarity = torch.matmul(local, overall)
            similarity = torch.matmul(similarity, local_weight).squeeze()
            # similarity : [query_len(1), query_len(2), query_len(3)]
            # similarity = torch.sum(similarity, dim=-1)
            # similarity : [query_len(1), query_len(2)]
            similarity_norm = torch.softmax(similarity, dim=-1)
            abs_pos = torch.matmul(similarity_norm, abs_pos)
            # abs_pos : [query_len, d_v]

            return abs_pos

        def training_step(self, query_len):
            return self.forward(query_len)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer]

    class PositionwiseFeedforwardLayer(pl.LightningModule):
        def __init__(self, d_model, d_ff, dropout_ratio):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True

            self.w_ff1 = nn.Linear(d_model, d_ff)
            nn.init.xavier_uniform_(self.w_ff1.weight.data)
            self.w_ff2 = nn.Linear(d_ff, d_model)
            nn.init.xavier_uniform_(self.w_ff2.weight.data)

            self.dropout = nn.Dropout(dropout_ratio)

        def forward(self, x):
            # with profiler.record_function("PositionwiseFeedforwardLayer"):
            # x: [batch_size, seq_len, d_model]
            x = self.dropout(torch.relu(self.w_ff1(x)))
            # x: [batch_size, seq_len, d_ff]
            x = self.w_ff2(x)
            # x: [batch_size, seq_len, d_model]
            return x

        def training_step(self, batch, batch_idx):
            x = batch
            return self.forward(x)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer] #, [lr_scheduler]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    TransformerEncoderLayer
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class TransformerEncoderLayer(pl.LightningModule):
        def __init__(self, d_k, d_v, d_model, n_heads, d_ff, dropout_ratio):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True

            self.self_attn_layer = MultiHeadAttentionLayer(d_k=d_k,
                                                           d_v=d_v,
                                                           d_model=d_model,
                                                           n_heads=n_heads,
                                                           dropout_ratio=dropout_ratio)
            self.self_attn_layer_norm = nn.LayerNorm(d_model)
            self.positionwise_ff_layer = PositionwiseFeedforwardLayer(d_model=d_model,
                                                                      d_ff=d_ff,
                                                                      dropout_ratio=dropout_ratio)
            self.positionwise_ff_layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout_ratio)

        def forward(self, src, src_mask):
            # src: [batch_size, src_len, d_model]
            # src_mask: [batch_size, src_len]

            attn_src, _ = self.self_attn_layer(src, src, src, src_mask)
            attn_add_norm_src = self.self_attn_layer_norm(src + self.dropout(attn_src))

            ff_src = self.positionwise_ff_layer(attn_add_norm_src)
            ff_add_norm_src = self.positionwise_ff_layer_norm(self.dropout(attn_add_norm_src) + ff_src)

            return ff_add_norm_src

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.forward(x, y)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer] #, [lr_scheduler]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    TransformerEncoder
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class TransformerEncoder(pl.LightningModule):
        def __init__(self, input_dim, d_k, d_v, d_model, n_layers, n_heads, d_ff, dropout_ratio, max_length=100):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True

            # self.pretrained_embedding = PretrainedEmbedding(SRC.vocab.stoi, d_model, filename='multi30k_src_glove.model')
            self.input_embedding = nn.Embedding(len(SRC.vocab.stoi), d_model)
            nn.init.xavier_uniform_(self.input_embedding.weight.data)

            '''since no <sos> <eos> are considered in max_len_sentence, we need to +2'''
            # self.pos_encoding = PositionalEncoding(max_len_sentence+2, hparams.d_model)

            self.layers = nn.ModuleList([TransformerEncoderLayer(d_k=d_k,
                                                                 d_v=d_v,
                                                                 d_model=d_model,
                                                                 n_heads=n_heads,
                                                                 d_ff=d_ff,
                                                                 dropout_ratio=dropout_ratio) for _ in range(n_layers)])
            self.dropout = nn.Dropout(dropout_ratio)
            # self.scale = torch.sqrt(torch.FloatTensor([d_k]))

        def forward(self, src, src_mask):
            batch_size = src.shape[0]
            src_len = src.shape[1]

            '''to map position index information'''

            # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)
            # src = self.dropout(self.tok_embedding(src))
            # pe = get_sinusoid_encoding_table(src_len, src.shape[2], self.device)
            # +positional encoding
            src = self.dropout(self.input_embedding(src))
            # src = self.dropout(self.pos_encoding(self.input_embedding(src)))
            # del pe
            # src: [batch_size, src_len, d_model]
            for layer in self.layers:
                src = layer(src, src_mask)

            return src

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.forward(x, y)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer] #, [lr_scheduler]


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    TransformerDecoderLayer
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class TransformerDecoderLayer(pl.LightningModule):
        def __init__(self, d_k, d_v, d_model, n_heads, d_ff, dropout_ratio):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True
            self.self_attn_layer = MultiHeadAttentionLayer(d_k=d_k,
                                                           d_v=d_v,
                                                           d_model=d_model,
                                                           n_heads=n_heads,
                                                           dropout_ratio=dropout_ratio)

            self.self_attn_layer_norm = nn.LayerNorm(d_model)

            self.enc_dec_attn_layer = MultiHeadAttentionLayer(d_k=d_k,
                                                              d_v=d_v,
                                                              d_model=d_model,
                                                              n_heads=n_heads,
                                                              dropout_ratio=dropout_ratio,
                                                              self_=False)

            self.enc_dec_attn_layer_norm = nn.LayerNorm(d_model)

            self.positionwise_ff_layer = PositionwiseFeedforwardLayer(d_model=d_model,
                                                                      d_ff=d_ff,
                                                                      dropout_ratio=dropout_ratio)
            self.positionwise_ff_layer_norm = nn.LayerNorm(d_model)

            self.dropout = nn.Dropout(dropout_ratio)

        def forward(self, trg, enc_src, trg_mask, src_mask):
            # trg: [batch_size, trg_len, d_model]
            # enc_src: [batch_size, src_len, d_model]
            # trg_mask: [batch_size, trg_len]
            # enc_mask: [batch_size, src_len]

            self_attn_trg, _ = self.self_attn_layer(trg, trg, trg, trg_mask)
            self_attn_add_norm_trg = self.self_attn_layer_norm(trg + self.dropout(self_attn_trg))
            enc_dec_attn_trg, attention = self.enc_dec_attn_layer(self_attn_add_norm_trg, enc_src, enc_src, src_mask)
            enc_dec_add_norm_trg = self.enc_dec_attn_layer_norm(self_attn_add_norm_trg + self.dropout(enc_dec_attn_trg))
            ff_trg = self.positionwise_ff_layer(enc_dec_add_norm_trg)
            ff_add_norm_trg = self.positionwise_ff_layer_norm(enc_dec_add_norm_trg + self.dropout(ff_trg))

            return ff_add_norm_trg, attention

        def training_step(self, batch, batch_idx):
            x, y, tm, sm = batch
            return self.forward(x, y, tm, sm)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer] #, [lr_scheduler]


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    TransformerDecoder
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class TransformerDecoder(pl.LightningModule):
        def __init__(self, output_dim, d_k, d_v, d_model, n_layers, n_heads, d_ff, dropout_ratio, max_length=100):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True

            # self.pretrained_embedding = PretrainedEmbedding(TRG.vocab.stoi, d_model, filename='multi30k_trg_glove.model')
            self.input_embedding = nn.Embedding(len(TRG.vocab.stoi), d_model)
            # nn.init.xavier_uniform_(self.input_embedding.weight.data)

            '''since no <sos> <eos> are considered in max_len_sentence, we need to +2'''
            # self.pos_encoding = PositionalEncoding(max_length, hparams.d_model)
            # self.pos_embedding = nn.Embedding(max_length, d_model)
            # nn.init.xavier_uniform_(self.pos_embedding.weight.data)

            self.layers = nn.ModuleList([TransformerDecoderLayer(d_k=d_k,
                                                                 d_v=d_v,
                                                                 d_model=d_model,
                                                                 n_heads=n_heads,
                                                                 d_ff=d_ff,
                                                                 dropout_ratio=dropout_ratio) for _ in range(n_layers)])
            self.affine = nn.Linear(d_model, output_dim)
            nn.init.xavier_uniform_(self.affine.weight.data)
            self.dropout = nn.Dropout(dropout_ratio)
            # self.scale = torch.sqrt(torch.FloatTensor([d_k]))

        def forward(self, trg, enc_src, trg_mask, src_mask):
            batch_size = trg.shape[0]
            trg_len = trg.shape[1]

            # pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1)
            # pos: [batch_size, trg_len]
            # trg = self.dropout(self.tok_embedding(trg))
            # trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
            # pe = get_sinusoid_encoding_table(trg_len, trg.shape[2], self.device)
            # trg = self.positional_encoding(trg)

            # '''+positional encoding'''
            # with torch.no_grad():
            #     trg += sinusoid_encoding_table[:trg_len, :]

            # del pe
            # trg: [batch_size, trg_len, d_model]
            trg = self.dropout(self.input_embedding(trg))
            # trg = self.dropout(self.pos_encoding(self.input_embedding(trg)))
            for layer in self.layers:
                trg, attention = layer(trg, enc_src, trg_mask, src_mask)

            output = self.affine(trg)
            # output: [batch_size, trg_len, output_len]
            return output, attention

        def training_step(self, batch, batch_idx):
            x, y, tm, sm = batch
            return self.forward(x, y, tm, sm)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            ''' 
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                    lr_lambda=lambda steps:(hparams.d_model**(-0.5))*min((steps+1)**(-0.5), (steps+1)*hparams.warmup_steps**(-1.5)),
                                                    last_epoch=-1)
            # lr_scheduler = {'scheduler':scheduler, 'name':'my_log'}
            lr_scheduler = {'scheduler':scheduler, 'name':'lr-base', 'interval':'step', 'frequency':1}
            '''

            return [optimizer] #, [lr_scheduler]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Transformer
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class Transformer(pl.LightningModule):
        def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, d_model, dropout_ratio):
            super().__init__()
            @property
            def automatic_optimization(self):
                return True
            self.encoder = encoder
            self.decoder = decoder
            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx

            self.dropout = nn.Dropout(dropout_ratio)
            # self.positional_encoding = PositionalEncoding(max_len_sentence+2, d_model)

        def make_src_mask(self, src):
            # src: [batch_size, src_len]
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            src_mask = src_mask.type_as(src)
            # src_mask: [batch_size, 1, 1, src_len]
            return src_mask

        def make_trg_mask(self, trg):
            # trg: [batch_size, trg_len]
            trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
            trg_pad_mask = trg_pad_mask.type_as(trg)
            # trg_pad_mask = [batch_size, 1, 1, trg_len]
            trg_len = trg.shape[1]
            trg_attn_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
            trg_attn_mask = trg_attn_mask.type_as(trg)
            # trg_attn_mask = [trg_len, trg_len]
            with torch.no_grad():
                trg_mask = trg_pad_mask & trg_attn_mask
            return trg_mask

        def forward(self, src, trg):

            # src: [batch_size, src_len]
            # trg: [batch_size, trg_len]

            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            # src_mask: [batch_size, 1, 1, src_len]
            # trg_mask: [batch_size, 1, trg_len, trg_len]
            # embed_src = self.dropout(self.pretrained_embedding(src))
            # pos_enc_src = self.positional_encoding(embed_src)
            enc_src = self.encoder(src, src_mask)
            # enc_src: [batch_size, src_len, d_model]
            # embed_trg = self.dropout(self.pretrained_embedding(trg))
            # pos_enc_trg = self.positional_encoding(embed_trg)
            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
            # output: [batch_size, trg_len, output_dim]
            # attention: [batch_size, n_heads, trg_len, src_len]
            return output, attention

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.forward(x, y)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            return [optimizer] #, [lr_scheduler]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    bleu score
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    from torchtext.data.metrics import bleu_score

    def show_bleu(data, SRC, TRG, model, device, logging = False, max_len=50):
        trgs = []
        pred_trgs = []
        index = 0

        for datum in data:
            src = vars(datum)['src']
            trg = vars(datum)['trg']

            pred_trg, _ = translate_sentence(src, SRC, TRG, model, device, max_len, logging=False)

            # remove <eos>
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

            index+=1
            if (index + 1) % 100 == 0 and logging:
                print(f'[{index+1}/{len(data)}]')
                print(f'pred: {pred_trg}')
                print(f'answer: {trg}')
        bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
        print(f'Total BLEU Score = {bleu*100:.2f}')
        sys.stdout.flush()

        # individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1,0,0,0])
        # individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0,1,0,0])
        # individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0,0,1,0])
        # individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0,0,0,1])
        #
        # cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1,0,0,0])
        # cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2,1/2,0,0])
        # cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3,1/3,1/3,0])
        # cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4,1/4,1/4,1/4])
        #
        # print(f'Individual BLEU1 score = {individual_bleu1_score * 100:.2f}')
        # print(f'Individual BLEU2 score = {individual_bleu2_score * 100:.2f}')
        # print(f'Individual BLEU3 score = {individual_bleu3_score * 100:.2f}')
        # print(f'Individual BLEU4 score = {individual_bleu4_score * 100:.2f}')
        #
        # print(f'Cumulative BLEU1 score = {cumulative_bleu1_score * 100:.2f}')
        # print(f'Cumulative BLEU2 score = {cumulative_bleu2_score * 100:.2f}')
        # print(f'Cumulative BLEU3 score = {cumulative_bleu3_score * 100:.2f}')
        # print(f'Cumulative BLEU4 score = {cumulative_bleu4_score * 100:.2f}')

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Label Smoothing
    refered to https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size
    """
    class LabelSmoothingLoss(pl.LightningModule):
        def __init__(self, classes, smoothing=0.0, dim=-1, weight = None, ignore_index=1):
            """if smoothing == 0, it's one-hot method
               if 0 < smoothing < 1, it's smooth method
            """
            super(LabelSmoothingLoss, self).__init__()
            @property
            def automatic_optimization(self):
                return True
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.weight = weight
            self.cls = classes
            self.dim = dim
            self.ignore_index = ignore_index

        def forward(self, pred, target):
            # with profiler.record_function("LabelSmoothingLoss"):
            assert 0 <= self.smoothing < 1
            result = None
            pred = pred.log_softmax(dim=self.dim)
            if self.weight is not None:
                pred = pred * self.weight.unsqueeze(0)

            with torch.enable_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
                # true_dist.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
                # true_len = len([i for i in target if i!=self.ignore_index])
            # return torch.sum(torch.sum(-true_dist * pred, dim=self.dim)) / true_len
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


        def training_step(self, batch, batch_idx, optimizer_idx):
            x, y = batch
            return self.forward(x, y)

    INPUT_DIM = len(SRC.vocab.stoi)
    OUTPUT_DIM = len(TRG.vocab.stoi)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    TrainModel (pytorch lightning)
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class TrainModel(pl.LightningModule):
        @property
        def automatic_optimization(self):
            return True

        def __init__(self):

            super(TrainModel, self).__init__()
            # self.automatic_optimization = False

            enc = TransformerEncoder(input_dim=INPUT_DIM,
                                     d_k=hparams.d_k,
                                     d_v=hparams.d_v,
                                     d_model=hparams.d_model,
                                     n_layers=hparams.n_decoder,
                                     n_heads=hparams.n_heads,
                                     d_ff=hparams.d_ff,
                                     dropout_ratio=hparams.dropout_ratio,
                                     max_length=max_len_sentence+2)

            dec = TransformerDecoder(output_dim=OUTPUT_DIM,
                                     d_k=hparams.d_k,
                                     d_v=hparams.d_v,
                                     d_model=hparams.d_model,
                                     n_layers=hparams.n_decoder,
                                     n_heads=hparams.n_heads,
                                     d_ff=hparams.d_ff,
                                     dropout_ratio=hparams.dropout_ratio,
                                     max_length=max_len_sentence+2)

            self.model = Transformer(encoder=enc,
                                     decoder=dec,
                                     src_pad_idx=SRC_PAD_IDX,
                                     trg_pad_idx=TRG_PAD_IDX,
                                     d_model=hparams.d_model,
                                     dropout_ratio=hparams.dropout_ratio)

            self.loss = LabelSmoothingLoss(smoothing=hparams.label_smoothing, classes=len(TRG.vocab.stoi),
                                           ignore_index=TRG_PAD_IDX)

            self.bleu_epoch_interval = 4
            # self.model.apply(utils.initalize_weights)
            '''
            pytorch lightning shows parameter summery already
            '''
            # print(f'The model has {utils.count_parameters(self.model):,} trainable parameters')
            # sys.stdout.flush()

        def forward(self, x, y):
            output, _ = self.model(x, y[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            return output

        def training_step(self, batch, batch_idx):
            x, y = batch
            output, _ = self.model(x, y[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            y = y[:, 1:].contiguous().view(-1)
            loss = self.loss(output, y)
            self.log("train_loss", loss, sync_dist=True, prog_bar=True)
            self.log('train_PPL', torch.exp(loss), sync_dist=True)

            # self.model.train()
            # self.manual_backward(loss)

            # optimizer.step()
            # scheduler.step()

            '''total loss in each epochs'''
            # global accumulate_loss
            # accumulate_loss += loss.item()
            # batches_loss = accumulate_loss / (batch_idx+1)
            # self.log('train_PPL', torch.exp(loss.float()), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
            # self.log('train_PPL', np.exp(batches_loss), sync_dist=True, prog_bar=True, logger=True)
            # self.log('train_loss', loss.float(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
            # self.log('train_loss', batches_loss, sync_dist=True, prog_bar=True, logger=True)
            # self.logger.experiment.add_scalar('Train/train_PPL', torch.exp(loss.float()), batch_idx)
            # self.logger.experiment.add_scalar('Train/train_loss', loss.float(), batch_idx)
            # tensorboard_logs = {'train_loss': loss, 'train_PPL': torch.exp(loss.float())}
            # return {'loss':batches_loss}
            return loss

        def training_epoch_end(self, outs):
            loss = torch.stack([outs[i]['loss'] for i in range(len(outs))]).mean()
            self.log("train_epochs_mean_loss", loss, sync_dist=True, prog_bar=True)
            self.log('train_epochs_mean_PPL', torch.exp(loss), sync_dist=True)
            if (self.current_epoch +1) % self.bleu_epoch_interval == 0:
                self.show_bleu_score(test, SRC, TRG, max_len=max_len_sentence+1)


        def validation_step(self, batch, batch_idx):
            x, y = batch
            output, _ = self.model(x, y[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            y = y[:, 1:].contiguous().view(-1)
            loss = self.loss(output, y)
            # self.log("val_loss", loss, sync_dist=True)
            # self.log('val_PPL', torch.exp(loss), sync_dist=True)
            # '''total loss in each epochs'''
            # global accumulate_loss
            # accumulate_loss += float(loss.item())
            # batches_loss = accumulate_loss / (batch_idx+1)
            # self.log('valid_PPL', torch.exp(loss.float()), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
            # self.log('valid_PPL', np.exp(batches_loss), sync_dist=True, prog_bar=True, logger=True)
            # self.log('valid_loss', loss.float(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
            # self.log('valid_loss', batches_loss, sync_dist=True, prog_bar=True, logger=True)
            # self.logger.experiment,
            # self.logger.experiment.add_scalar('Valid/valid_PPL', torch.exp(loss.float()), batch_idx)
            # self.logger.experiment.add_scalar('Valid/valid_loss', loss.float(), batch_idx)
            return loss

        def validation_epoch_end(self, outs):
            loss = torch.stack(outs).mean()
            self.log("val_loss", loss, sync_dist=True)
            self.log('val_PPL', torch.exp(loss), sync_dist=True)
            # self.show_bleu_score(test, SRC, TRG, max_len=max_len_sentence+1)

        def configure_optimizers(self):
            # warmup_steps = 4000
            optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-3, lr=hparams.learning_rate, amsgrad=True)
            '''
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                    lr_lambda=lambda steps:(hparams.d_model**(-0.5))*min((steps+1)**(-0.5), (steps+1)*hparams.warmup_steps**(-1.5)),
                                                    last_epoch=-1)
            # lr_scheduler = {'scheduler':scheduler, 'name':'my_log'}
            lr_scheduler = {'scheduler':scheduler, 'name':'lr-base', 'interval':'step', 'frequency':1}
			'''
            return [optimizer] #, [lr_scheduler]

        def get_progress_bar_dict(self):
            tqdm_dict = super().get_progress_bar_dict()
            if 'v_num' in tqdm_dict:
                del tqdm_dict['v_num']
            return tqdm_dict

        def translate_sentences(self, sentence, SRC, TRG, max_len=max_len_sentence, logging=False):

            self.model.eval()

            if isinstance(sentence, str):
                tokenizer = spacy.load('de_core_news_sm')
                tokens = [token.text.lower() for token in tokenizer(sentence)]
            else:
                tokens = [token.lower() for token in sentence]

            '''put <sos> in the first, <eos> in the end.'''
            tokens = [SRC.init_token] + tokens + [SRC.eos_token]
            '''convert to indexes'''
            src_indexes = [SRC.vocab.stoi[token] for token in tokens]

            if logging:
                print(f'src tokens : {tokens}')
                print(f'src indexes : {src_indexes}')

            src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)

            src_pad_mask = self.model.make_src_mask(src_tensor)

            with torch.no_grad():
                enc_src = self.model.encoder(src_tensor, src_pad_mask)

            '''always start with first token'''
            trg_indexes = [TRG.vocab.stoi[TRG.init_token]]

            for i in range(max_len):
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
                trg_mask = self.model.make_src_mask(trg_tensor)

                with torch.no_grad():
                    output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_pad_mask)

                # output: [batch_size, trg_len, output_dim]
                pred_token = output.argmax(2)[:, -1].item()
                trg_indexes.append(pred_token)

                if pred_token == TRG.vocab.stoi[TRG.eos_token]:
                    break

            trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]

            return trg_tokens[1:], attention

        def show_bleu_score(self, data, SRC, TRG, logging = False, max_len=max_len_sentence):
            trgs = []
            pred_trgs = []
            index = 0

            for datum in data[:100]:
                src = vars(datum)['src']
                trg = vars(datum)['trg']

                pred_trg, _ = self.translate_sentences(src, SRC, TRG, max_len, logging=False)

                # remove <eos>
                pred_trg = pred_trg[:-1]

                pred_trgs.append(pred_trg)
                trgs.append([trg])

                index += 1
                if (index + 1) % 100 == 0 and logging:
                    print(f'[{index + 1}/{len(data)}]')
                    print(f'pred: {pred_trg}')
                    print(f'answer: {trg}')

            bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
            self.log('bleu_score', bleu * 100, sync_dist=True)
            print(f'Total BLEU Score = {bleu * 100:.2f}')
            sys.stdout.flush()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    custom callback (pytorch lightning)
    from https://github.com/PyTorchLightning/pytorch-lightning/issues/2534#issuecomment-674582085
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    class CheckpointEveryNSteps(pl.Callback):
        """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
        """

        def __init__(
            self,
            save_batch_frequency,
            prefix="N-Step-Checkpoint",
            use_modelcheckpoint_filename=False,
        ):
            """
            Args:
                save_batch_frequency: how often to save in batches
                prefix: add a prefix to the name, only used if
                    use_modelcheckpoint_filename=False
                use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                    default filename, don't use ours.
            """
            self.save_batch_frequency = save_batch_frequency
            self.prefix = prefix
            self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

        # def configure_ddp(self, model, device_ids):
        #     model = LightningDistributedDataParallel(model, device_ids, find_unused_parameters=False)
        #     return model

        def on_batch_end(self, trainer: pl.Trainer, _):
            """ Check if we should save a checkpoint after every train batch """
            epoch = trainer.current_epoch
            # trainer.model.show_bleu_score(test, SRC, TRG, max_len=max_len_sentence)
            global_step = trainer.global_step + 1
            # print('global_step: ', global_step, 'frequency: ', self.save_batch_frequency)
            # sys.stdout.flush()
            if global_step % self.save_batch_frequency == 0:
                if self.use_modelcheckpoint_filename:
                    filename = trainer.checkpoint_callback.filename
                else:
                    filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
                    ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
                    trainer.save_checkpoint(ckpt_path)
                # trainer.show_bleu_score(test, SRC, TRG, max_len=max_len_sentence)
                # trainer.run_evaluation()
                # trainer.model.show_bleu_score(test, SRC, TRG)
        # def on_epoch_end(self, trainer: pl.Trainer, _):
        #     global accumulate_loss
        #     accumulate_loss=0

    '''set argument parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help='the number of gpus')
    args = parser.parse_args()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Training (pytorch lightning)
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''declare model, callbacks, monitors'''
    model = TrainModel()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    batches_for_1epoch = int(math.ceil(len(train.examples)//hparams.batch_size))
    save_total_num = 10
    nstep_check_callback = CheckpointEveryNSteps(save_batch_frequency=(batches_for_1epoch*hparams.n_epochs)//save_total_num)


    '''accumulate settings'''
    accumul_num = 16 
    '''check interval settings'''
    check_interval = 10
    val_check_interval = int(batches_for_1epoch // check_interval)

    '''print setting info'''
    print('validation_check_interval: ', val_check_interval, ' batch_steps')
    print('save_interval: ',  (batches_for_1epoch*hparams.n_epochs)//save_total_num, ' batch_steps')
    sys.stdout.flush()

    if device.type=='cpu':
        trainer = pl.Trainer(max_epochs=hparams.n_epochs,
                             callbacks=[nstep_check_callback],
                             val_check_interval=val_check_interval,
                             deterministic=True,
                             logger=logger,
                             flush_logs_every_n_steps=1,
                             log_every_n_steps=1,
                             progress_bar_refresh_rate=50)
    else:
        trainer = pl.Trainer(gpus=args.gpus,
                             max_epochs=hparams.n_epochs,
                             callbacks=[nstep_check_callback],
                             val_check_interval=val_check_interval,
                             accumulate_grad_batches=accumul_num,
                             deterministic=True,
                             accelerator="ddp",
                             logger=logger,
                             flush_logs_every_n_steps=1,
                             log_every_n_steps=1,
                             progress_bar_refresh_rate=1000,
                             plugins=DDPPlugin(find_unused_parameters=False),
                             precision=16)
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    program_loop()
# print(prof.key_averages(group_by_stack_n=3).table(sort_by='self_cpu_time_total', row_limit=10))
