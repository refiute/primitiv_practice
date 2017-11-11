#!/usr/bin/env python3
# coding: utf-8

import sys
import random
import math

from primitiv import Device, Parameter, Graph, Trainer
from primitiv import devices as D
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import trainers as T

from lstm import LSTM

class EncoderDecoder(object):
    def __init__(self, name, src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_rate):
        self.name_ = name
        self.embed_size_ = embed_size
        self.hidden_size_ = hidden_size
        self.dropout_rate_ = dropout_rate
        self.psrc_lookup_ = Parameter([embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup_ = Parameter([embed_size, trg_vocab_size], I.XavierUniform())
        self.pwfbw_ = Parameter([2*hidden_size, hidden_size], I.XavierUniform())
        self.pwhw_ = Parameter([hidden_size, hidden_size], I.XavierUniform())
        self.pwwe_ = Parameter([hidden_size], I.XavierUniform())
        self.pwhj_ = Parameter([embed_size, hidden_size], I.XavierUniform())
        self.pbj_ = Parameter([embed_size], I.Constant(0))
        self.pwjy_ = Parameter([trg_vocab_size, embed_size], I.XavierUniform())
        self.pby_ = Parameter([trg_vocab_size], I.Constant(0))
        self.src_fw_lstm_ = LSTM(name+"_src_fw_lstm", embed_size, hidden_size)
        self.src_bw_lstm_ = LSTM(name+"_src_bw_lstm", embed_size, hidden_size)
        self.trg_lstm_ = LSTM(name+"_trg_lstm", embed_size+hidden_size*2, hidden_size)

    # Loads all parameters.
    @staticmethod
    def load(name, prefix):
        encdec = EncoderDecoder.__new__(EncoderDecoder)
        encdec.name_ = name
        encdec.psrc_lookup_ = Parameter.load(prefix+name+"_src_lookup.param")
        encdec.ptrg_lookup_ = Parameter.load(prefix+name+"_trg_lookup.param")
        encdec.pwfbw_ = Parameter.load(prefix+name+"_wfbw.param")
        encdec.pwhw_ = Parameter.load(prefix+name+"_whw.param")
        encdec.pwwe_ = Parameter.load(prefix+name+"_wwe.param")
        encdec.pwhj_ = Parameter.load(prefix+name+"_whj.param")
        encdec.pbj_ = Parameter.load(prefix+name+"_bj.param")
        encdec.pwjy_ = Parameter.load(prefix+name+"_wjy.param")
        encdec.pby_ = Parameter.load(prefix+name+"_by.param")
        encdec.src_fw_lstm_ = LSTM.load(name+"_src_fw_lstm", prefix)
        encdec.src_bw_lstm_ = LSTM.load(name+"_src_bw_lstm", prefix)
        encdec.trg_lstm_ = LSTM.load(name+"_trg_lstm", prefix)
        encdec.embed_size_ = encdec.pbj_.shape()[0]
        encdec.hidden_size_ = encdec.pwhw_.shape()[0]
        with open(prefix+name+".config", "r", encoding="utf-8") as f:
            encdec.dropout_rate_ = float(f.readline())
        return encdec

    # Saves all parameters
    def save(self, prefix):
        self.psrc_lookup_.save(prefix+self.name_+"_src_lookup.param")
        self.ptrg_lookup_.save(prefix+self.name_+"_trg_lookup.param")
        self.pwfbw_.save(prefix+self.name_+"_wfbw.param")
        self.pwhw_.save(prefix+self.name_+"_whw.param")
        self.pwwe_.save(prefix+self.name_+"_wwe.param")
        self.pwhj_.save(prefix+self.name_+"_whj.param")
        self.pbj_.save(prefix+self.name_+"_bj.param")
        self.pwjy_.save(prefix+self.name_+"_wjy.param")
        self.pby_.save(prefix+self.name_+"_by.param")
        self.src_fw_lstm_.save(prefix)
        self.src_bw_lstm_.save(prefix)
        self.trg_lstm_.save(prefix)
        with open(prefix+self.name_+".config", "w", encoding="utf-8") as f:
            print(self.dropout_rate_, file=f)

    # Adds parameters to the trainer
    def register_training(self, trainer):
        trainer.add_parameter(self.psrc_lookup_)
        trainer.add_parameter(self.ptrg_lookup_)
        trainer.add_parameter(self.pwfbw_)
        trainer.add_parameter(self.pwhw_)
        trainer.add_parameter(self.pwwe_)
        trainer.add_parameter(self.pwhj_)
        trainer.add_parameter(self.pbj_)
        trainer.add_parameter(self.pwjy_)
        trainer.add_parameter(self.pby_)
        self.src_fw_lstm_.register_training(trainer)
        self.src_bw_lstm_.register_training(trainer)
        self.trg_lstm_.register_training(trainer)

    def encode(self, src_batch, train):
        # Embedding lookup.
        src_lookup = F.parameter(self.psrc_lookup_)
        e_list = []
        for x in src_batch:
            e = F.pick(src_lookup, x, 1)
            e = F.dropout(e, self.dropout_rate_, train)
            e_list.append(e)

        # Forward encoding
        self.src_fw_lstm_.init()
        f_list = []
        for e in e_list:
            f = self.src_fw_lstm_.forward(e)
            f = F.dropout(f, self.dropout_rate_, train)
            f_list.append(f)

        # Backward encoding
        self.src_bw_lstm_.init()
        b_list = []
        for e in reversed(e_list):
            b = self.src_bw_lstm_.forward(e)
            b = F.dropout(b, self.dropout_rate_, train)
            b_list.append(b)
        b_list.reverse()

        # Concatenates RNN states.
        self.fb_list = [F.concat([f_list[i], b_list[i]], 0) for i in range(len(src_batch))]
        self.concat_fb = F.concat(self.fb_list, 1)
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
        self.trg_lstm_.init()

    # One step decoding.
    def decode_step(self, trg_words, train):
        b = self.whw_ @ self.trg_lstm_.get_h()
        b = F.transpose(F.broadcast(b, 1, len(self.fb_list)))
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
