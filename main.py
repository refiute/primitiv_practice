#!/usr/bin/env python3
# coding: utf-8

import sys
import random
import math

from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

from primitiv import Device, Graph, Optimizer
from primitiv import devices as D
from primitiv import optimizers as O

from utils import (
    make_vocab, load_corpus, load_corpus_ref, count_labels, make_batch,
    save_ppl, make_inv_vocab, line_to_sent, load_ppl
)
from bleu import get_bleu_stats, calculate_bleu
from bahdanau_encdec import EncoderDecoder

SRC_TRAIN_FILE = "data/train.en"
TRG_TRAIN_FILE = "data/train.ja"
SRC_VALID_FILE = "data/dev.en"
TRG_VALID_FILE = "data/dev.ja"
SRC_TEST_FILE = "data/test.en"
REF_TEST_FILE = "data/test.ja"

# Training encode decode model.
def train(encdec, optimizer, args, best_valid_ppl):
    prefix = args.model
    max_epoch = args.epoch
    batch_size = args.minibatch

    # Registers all parameters to the optimizer.
    optimizer.add(encdec)

    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, args.src_vocab)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, args.trg_vocab)
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
    for epoch in range(max_epoch):
        # Computation graph.
        g = Graph()
        Graph.set_default(g)

        print("epoch %d/%d:" % (epoch + 1, max_epoch))
        print("  learning rate scale = %.4e" % optimizer.get_learning_rate_scaling())

        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0.
        for ofs in range(0, num_train_sents, batch_size):
            print("%d" % ofs, end="\r", flush=True)

            batch_ids = train_ids[ofs:min(ofs+args.minibatch, num_train_sents)]
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
        for ofs in range(0, num_valid_sents, batch_size):
            print("%d" % ofs, end="\r", flush=True)

            batch_ids = valid_ids[ofs:min(ofs+batch_size, num_valid_sents)]
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
        for ofs in range(0, num_test_sents, batch_size):
            print("%d" % ofs, end="\r", flush=True)

            src_batch = test_src_corpus[ofs:min(ofs + batch_size, num_test_sents)]
            ref_batch = test_ref_corpus[ofs:min(ofs + batch_size, num_test_sents)]

            hyp_ids = test_batch(encdec, src_vocab, trg_vocab,
                                 src_batch, args.generation_limit)
            for hyp_line, ref_line in zip(hyp_ids, ref_batch):
                for k, v in get_bleu_stats(ref_line[1:-1], hyp_line).items():
                    stats[k] += v

        bleu = calculate_bleu(stats)
        print("  test BLEU = %.2f" % (100 * bleu))

        # Saves best model/optimizer.
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  saving model/optimizer ... ", end="", flush=True)
            encdec.save(prefix+".model")
            optimizer.save(prefix+".optimizer")
            save_ppl(prefix+".valid_ppl", best_valid_ppl)
            print("done.")
        else:
            # Learning rate decay by 1/sqrt(2)
            new_scale = .7071 * optimizer.get_learning_rate_scaling()
            optimizer.set_learning_rate_scaling(new_scale)

def test_batch(encdec, src_vocab, trg_vocab, lines, generation_limit):
    g = Graph()
    Graph.set_default(g)

    src_batch = make_batch(lines, list(range(len(lines))), src_vocab)
    encdec.encode(src_batch, False)

    trg_ids = [np.array([trg_vocab["<bos>"]] * len(lines))]
    eos_id = trg_vocab["<eos>"]
    eos_ids = np.array([eos_id] * len(lines))
    while (trg_ids[-1] != eos_ids).any():
        if len(trg_ids) > generation_limit + 1:
            print("Warning: Sentence generation did not finish in", generation_limit,
                  "iterations.", file=sys.stderr)
            trg_ids.append(eos_ids)
            break
        y = encdec.decode_step(trg_ids[-1], False)
        trg_ids.append(np.array(y.argmax(0)).T)

    return [hyp[:np.where(hyp == eos_id)[0][0]] for hyp in np.array(trg_ids[1:]).T]

# Generates translation by consuming stdin.
def test(encdec, args):
    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, args.src_vocab)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, args.trg_vocab)
    inv_trg_vocab = make_inv_vocab(trg_vocab)

    for line in sys.stdin:
        sent = [line_to_sent(line.strip(), src_vocab)]
        trg_ids = test_batch(encdec, src_vocab, trg_vocab,
                             sent, args.generation_limit)[0]
        # Prints the result.
        print(" ".join(inv_trg_vocab[wid] for wid in trg_ids))


def get_arguments():

    src_vocab = 4000
    trg_vocab = 5000
    embed_size = 512
    hidden_size = 512
    epoch = 30
    minibatch_size = 64
    generation_limit = 32
    dropout = 0.5

    parser = ArgumentParser()
    parser.add_argument("mode", help="'train', 'resume', or 'test'")
    parser.add_argument("model", help='model file prefix')
    parser.add_argument("--gpu", default=-1, metavar='INT', type=int,
                        help='GPU device ID to be used (default: %(default)d [use CPU])')
    parser.add_argument("--src-vocab", default=src_vocab, metavar='INT', type=int,
                        help="source vocabulary size (default: %(default)d)")
    parser.add_argument("--trg-vocab", default=trg_vocab, metavar='INT', type=int,
                        help="target vocabulary size (default: %(default)d)")
    parser.add_argument("--embed", default=embed_size, metavar='INT', type=int,
                        help="embedding layer size (default: %(default)d)")
    parser.add_argument("--hidden", default=hidden_size, metavar='INT', type=int,
                        help="hidden layer size (default: %(default)d)")
    parser.add_argument("--epoch", default=epoch, metavar='INT', type=int,
                        help="number of training epoch (default: %(default)d)")
    parser.add_argument("--minibatch", default=minibatch_size, metavar="INT", type=int,
                        help="minibatch size (default: %(default)d)")
    parser.add_argument("--generation-limit", default=generation_limit, metavar="INT", type=int,
                        help="maximum number of words to be generated for test input (default: %(default)d)")
    parser.add_argument("--dropout", default=dropout_rate, metavar="FLOAT", type=flaot, help="dropout rate")

    args = parser.parse_args()
    try:
        if args.mode not in ("train", "resume", "test"):
            raise ValueError("you must set mode = 'train', 'resume', or 'test'")
        if args.gpu < 0:
            raise ValueError("you must set --gpu >= 0")
        if args.src_vocab < 1:
            raise ValueError("you must set --src-vocab >= 1")
        if args.trg_vocab < 1:
            raise ValueError("you must set --trg-vocab >= 1")
        if args.embed < 1:
            raise ValueError("you must set --embed >= 1")
        if args.hidden < 1:
            raise ValueError("you must set --hidden >= 1")
        if args.epoch < 1:
            raise ValueError("you must set --epoch >= 1")
        if args.minibatch < 1:
            raise ValueError("you must set --minibatch >= 1")
        if args.generation_limit < 1:
            raise ValueError("you must set --generation-limit >= 1")
        if args.dropout < 0 or args.dropout > 1:
            raise ValueError("you must set --dropout in [0, 1]")
    except Exception as ex:
        parser.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    for (key, val) in vars(args).items():
        print("%s: %s" % (key, val))

    return args

def main():
    args = get_arguments()

    print("initializing device ... ", end="", file=sys.stderr, flush=True)
    dev = D.Naive() if args.gpu < 0 else D.CUDA(args.gpu)
    Device.set_default(dev)
    print("done.", file=sys.stderr)

    mode = args.mode
    prefix = args.model
    if mode == "train":
        encdec = EncoderDecoder(args.dropout)
        encdec.init(args.src_vocab, args.trg_vocab, args.embed, args.hidden)
        optimizer = O.Adam()
        optimizer.set_weight_decay(1e-6)
        optimizer.set_gradient_clipping(5)
        train(encdec, optimizer, args, 1e10)
    elif mode == "resume":
        print("loading model/optimizer ... ", end="", file=sys.stderr, flush=True)
        encdec = EncoderDecoder(args.dropout)
        encdec.load(prefix+".model")
        optimizer = O.Adam()
        optimizer.load(prefix + ".optimizer")
        valid_ppl = load_ppl(prefix + ".valid_ppl")
        print("done.", file=sys.stderr)
        train(encdec, optimizer, args, valid_ppl)
    else:
        print("loading model ... ", end="", file=sys.stderr, flush=True)
        encdec = EncoderDecoder(args.dropout)
        encdec.load(prefix+".model")
        print("done.", file=sys.stderr)
        test(encdec, args)

if __name__ == "__main__":
    main()
