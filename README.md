# primitiv practice

[primitiv/primitiv-python](https://github.com/primitiv/primitiv-python) の練習用

[Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., ICLR'15)](https://arxiv.org/abs/1409.0473)を参考に
[odashi/small_parallel_enja](https://github.com/odashi/small_parallel_enja), beam searchなし, BLEU値が37ぐらい


## Usage
### Install primitiv and python-primitiv
See [primitiv/primitiv-python](https://github.com/primitiv/primitiv-python)

### training
```
git clone [this repository]
cd primitiv_practice
git clone https://github.com/odashi/small_parallel_enja data
mkdir model
python main.py [--gpu GPUID] train model/small_enja
python main.py [--gpu GPUID] test model/small_enja
```
