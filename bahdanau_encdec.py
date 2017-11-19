#!/usr/bin/env python3
# coding: utf-8

from primitiv import Parameter, Model, Shape
from primitiv import operators as F
from primitiv import initializers as I

from lstm import LSTM

class EncoderDecoder(Model):
    def __init__(self, dropout_rate):
        super().__init__()

        self.dropout_rate_ = dropout_rate

        self.psrc_lookup_ = Parameter()
        self.ptrg_lookup_ = Parameter()
        self.pwfbw_ = Parameter()
        self.pwhw_ = Parameter()
        self.pwwe_ = Parameter()
        self.pwhj_ = Parameter()
        self.pbj_ = Parameter()
        self.pwjy_ = Parameter()
        self.pby_ = Parameter()
        self.src_fw_lstm_ = LSTM()
        self.src_bw_lstm_ = LSTM()
        self.trg_lstm_ = LSTM()

        self.add_parameter("src_lookup", self.psrc_lookup_)
        self.add_parameter("trg_lookup", self.ptrg_lookup_)
        self.add_parameter("wfbw", self.pwfbw_)
        self.add_parameter("whw", self.pwhw_)
        self.add_parameter("wwe", self.pwwe_)
        self.add_parameter("whj", self.pwhj_)
        self.add_parameter("bj", self.pbj_)
        self.add_parameter("wjy", self.pwjy_)
        self.add_parameter("by", self.pby_)
        self.add_submodel("src_fw_lstm", self.src_fw_lstm_)
        self.add_submodel("src_bw_lstm", self.src_bw_lstm_)
        self.add_submodel("trg_lstm_", self.trg_lstm_)

    def init(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size):
        self.psrc_lookup_.init([embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup_.init([embed_size, trg_vocab_size], I.XavierUniform())
        self.pwfbw_.init([2*hidden_size, hidden_size], I.XavierUniform())
        self.pwhw_.init([hidden_size, hidden_size], I.XavierUniform())
        self.pwwe_.init([hidden_size], I.XavierUniform())
        self.pwhj_.init([embed_size, hidden_size], I.XavierUniform())
        self.pbj_.init([embed_size], I.Constant(0))
        self.pwjy_.init([trg_vocab_size, embed_size], I.XavierUniform())
        self.pby_.init([trg_vocab_size], I.Constant(0))
        self.src_fw_lstm_.init(embed_size, hidden_size)
        self.src_bw_lstm_.init(embed_size, hidden_size)
        self.trg_lstm_.init(embed_size+hidden_size*2, hidden_size)

    def encode(self, src_batch, train):
        # Embedding lookup.
        src_lookup = F.parameter(self.psrc_lookup_)
        e_list = []
        for x in src_batch:
            e = F.pick(src_lookup, x, 1)
            e = F.dropout(e, self.dropout_rate_, train)
            e_list.append(e)

        # Forward encoding
        self.src_fw_lstm_.reset()
        f_list = []
        for e in e_list:
            f = self.src_fw_lstm_.forward(e)
            f = F.dropout(f, self.dropout_rate_, train)
            f_list.append(f)

        # Backward encoding
        self.src_bw_lstm_.reset()
        b_list = []
        for e in reversed(e_list):
            b = self.src_bw_lstm_.forward(e)
            b = F.dropout(b, self.dropout_rate_, train)
            b_list.append(b)
        b_list.reverse()

        # Concatenates RNN states.
        fb_list = [F.concat([f_list[i], b_list[i]], 0) for i in range(len(src_batch))]
        self.concat_fb = F.concat(fb_list, 1)
        self.t_concat_fb = F.transpose(self.concat_fb)

        # Initializes decode states.
        self.wfbw_ = F.parameter(self.pwfbw_)
        self.whw_ = F.parameter(self.pwhw_)
        self.wwe_ = F.parameter(self.pwwe_)
        self.trg_lookup_ = F.parameter(self.ptrg_lookup_)
        self.whj_ = F.parameter(self.pwhj_)
        self.bj_ = F.parameter(self.pbj_)
        self.wjy_ = F.parameter(self.pwjy_)
        self.by_ = F.parameter(self.pby_)
        self.trg_lstm_.reset()

    # One step decoding.
    def decode_step(self, trg_words, train):
        sentence_len = self.concat_fb.shape()[1]

        b = self.whw_ @ self.trg_lstm_.get_h()
        b = F.reshape(b, Shape([1, b.shape()[0]]))
        b = F.broadcast(b, 0, sentence_len)
        x = F.tanh(self.t_concat_fb @ self.wfbw_ + b)
        atten_prob = F.softmax(x @ self.wwe_, 0)
        c = self.concat_fb @ atten_prob

        e = F.pick(self.trg_lookup_, trg_words, 1)
        e = F.dropout(e, self.dropout_rate_, train)

        h = self.trg_lstm_.forward(F.concat([e, c], 0))
        h = F.dropout(h, self.dropout_rate_, train)
        j = F.tanh(self.whj_ @ h + self.bj_)
        return self.wjy_ @ j + self.by_

    # Calculates the loss function over given target sentences.
    def loss(self, trg_batch, train):
        losses = []
        for i in range(len(trg_batch)-1):
            y = self.decode_step(trg_batch[i], train)
            loss = F.softmax_cross_entropy(y, trg_batch[i+1], 0)
            losses.append(loss)
        return F.batch.mean(F.sum(losses))
