#!/usr/bin/env python3
# coding: utf-8

import sys
import random
import math

from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

from primitiv import Device, Parameter, Graph, Optimizer
from primitiv import devices as D
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import optimizers as O

from utils import (
    make_vocab, load_corpus, load_corpus_ref, count_labels, make_batch,
    save_ppl, make_inv_vocab, line_to_sent, load_ppl
)
from bleu import get_bleu_stats, calculate_bleu
from bahdanau_encdec import EncoderDecoder

SRC_VOCAB_SIZE = 4000
TRG_VOCAB_SIZE = 5000
NUM_EMBED_UNITS = 512
NUM_HIDDEN_UNITS = 512
BATCH_SIZE = 64
MAX_EPOCH = 30
GENERATION_LIMIT = 32
DROPOUT_RATE = 0.5
SRC_TRAIN_FILE = "data/train.en"
TRG_TRAIN_FILE = "data/train.ja"
SRC_VALID_FILE = "data/dev.en"
TRG_VALID_FILE = "data/dev.ja"
SRC_TEST_FILE = "data/test.en"
REF_TEST_FILE = "data/test.ja"

# Training encode decode model.
def train(encdec, optimizer, prefix, best_valid_ppl):
    # Registers all parameters to the optimizer.

    # optimizer.add_model(encdec)
    for param in encdec.get_trainable_parameters().values():
        optimizer.add_parameter(param)

    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)
    print("#src_vocab:", len(src_vocab))
    print("#trg_vocab:", len(trg_vocab))

    # Loads all corpus
    train_src_corpus = load_corpus(SRC_TRAIN_FILE, src_vocab)
    train_trg_corpus = load_corpus(TRG_TRAIN_FILE, trg_vocab)
    valid_src_corpus = load_corpus(SRC_VALID_FILE, src_vocab)
    valid_trg_corpus = load_corpus(TRG_VALID_FILE, trg_vocab)
    test_src_corpus = load_corpus(SRC_TEST_FILE, src_vocab)
    test_ref_corpus = load_corpus_ref(REF_TEST_FILE, trg_vocab)
    num_train_sents = len(train_trg_corpus)
    num_valid_sents = len(valid_trg_corpus)
    num_test_sents = len(test_ref_corpus)
    num_train_labels = count_labels(train_trg_corpus)
    num_valid_labels = count_labels(valid_trg_corpus)
    print("train:", num_train_sents, "sentences,", num_train_labels, "labels")
    print("valid:", num_valid_sents, "sentences,", num_valid_labels, "labels")

    # Sentence IDs
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    # Train/valid loop.
    for epoch in range(MAX_EPOCH):
        # Computation graph.
        g = Graph()
        Graph.set_default(g)

        print("epoch %d/%d:" % (epoch + 1, MAX_EPOCH))
        print("  learning rate scale = %.4e" % optimizer.get_learning_rate_scaling())

        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0.
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = train_ids[ofs:min(ofs+BATCH_SIZE, num_train_sents)]
            src_batch = make_batch(train_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(train_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, True)
            loss = encdec.loss(trg_batch, True)
            train_loss += loss.to_float() * len(batch_ids)

            optimizer.reset_gradients()
            loss.backward()
            optimizer.update()

        train_ppl = math.exp(train_loss / num_train_labels)
        print("  train PPL = %.4f" % train_ppl)

        # Validation.
        valid_loss = 0.
        for ofs in range(0, num_valid_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = valid_ids[ofs:min(ofs+BATCH_SIZE, num_valid_sents)]
            src_batch = make_batch(valid_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(valid_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, False)
            loss = encdec.loss(trg_batch, False)
            valid_loss += loss.to_float() * len(batch_ids)

        valid_ppl = math.exp(valid_loss/num_valid_labels)
        print("  valid PPL = %.4f" % valid_ppl)

        # Calculates test BLEU.
        stats = defaultdict(int)
        for ofs in range(0, num_test_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            src_batch = test_src_corpus[ofs:min(ofs + BATCH_SIZE, num_test_sents)]
            ref_batch = test_ref_corpus[ofs:min(ofs + BATCH_SIZE, num_test_sents)]

            hyp_ids = test_batch(encdec, src_vocab, trg_vocab, src_batch)
            for hyp_line, ref_line in zip(hyp_ids, ref_batch):
                for k, v in get_bleu_stats(ref_line[1:-1], hyp_line).items():
                    stats[k] += v

        bleu = calculate_bleu(stats)
        print("  test BLEU = %.2f" % (100 * bleu))

        # Saves best model/optimizer.
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  saving model/optimizer ... ", end="")
            sys.stdout.flush()
            encdec.save(prefix+".model")
            optimizer.save(prefix+".optimizer")
            save_ppl(prefix+".valid_ppl", best_valid_ppl)
            print("done.")
        else:
            # Learning rate decay by 1/sqrt(2)
            new_scale = .7071 * optimizer.get_learning_rate_scaling()
            optimizer.set_learning_rate_scaling(new_scale)

def test_batch(encdec, src_vocab, trg_vocab, lines):
    g = Graph()
    Graph.set_default(g)

    src_batch = make_batch(lines, list(range(len(lines))), src_vocab)
    encdec.encode(src_batch, False)

    trg_ids = [np.array([trg_vocab["<bos>"]] * len(lines))]
    eos_id = trg_vocab["<eos>"]
    eos_ids = np.array([eos_id] * len(lines))
    while (trg_ids[-1] != eos_ids).any():
        if len(trg_ids) > GENERATION_LIMIT + 1:
            print("Warning: Sentence generation did not finish in", GENERATION_LIMIT,
                  "iterations.", file=sys.stderr)
            trg_ids.append(eos_ids)
            break
        y = encdec.decode_step(trg_ids[-1], False)
        trg_ids.append(np.array(y.argmax(0)).T)

    return [hyp[:np.where(hyp == eos_id)[0][0]] for hyp in np.array(trg_ids[1:]).T]

# Generates translation by consuming stdin.
def test(encdec):
    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)

    for line in sys.stdin:
        sent = [line_to_sent(line.strip(), src_vocab)]
        trg_ids = test_batch(encdec, src_vocab, trg_vocab, sent)[0]
        # Prints the result.
        print(" ".join(inv_trg_vocab[wid] for wid in trg_ids))

def main():
    parser = ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("model_prefix")
    args = parser.parse_args()

    mode = args.mode
    prefix = args.model_prefix
    print("mode:", mode, file=sys.stderr)
    print("prefix:", prefix, file=sys.stderr)

    if mode not in ("train", "resume", "test"):
        print("unknown mode:", mode, file=sys.stderr)
        return

    print("initializing device ... ", end="", file=sys.stderr)
    sys.stderr.flush()

    # dev = D.Naive()
    dev = D.CUDA(0)
    Device.set_default(dev)
    print("done.", file=sys.stderr)

    if mode == "train":
        encdec = EncoderDecoder(DROPOUT_RATE)
        encdec.init(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, NUM_EMBED_UNITS, NUM_HIDDEN_UNITS)
        optimizer = O.Adam()
        optimizer.set_weight_decay(1e-6)
        optimizer.set_gradient_clipping(5)
        train(encdec, optimizer, prefix, 1e10)
    elif mode == "resume":
        print("loading model/optimizer ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = EncoderDecoder(DROPOUT_RATE)
        encdec.load(prefix+".model")
        optimizer = O.Adam()
        optimizer.load(prefix + ".optimizer")
        valid_ppl = load_ppl(prefix + ".valid_ppl")
        print("done.", file=sys.stderr)
        train(encdec, optimizer, prefix, valid_ppl)
    else:
        print("loading model ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = EncoderDecoder(DROPOUT_RATE)
        encdec.load(prefix+".model")
        print("done.", file=sys.stderr)
        test(encdec)

if __name__ == "__main__":
    main()
