# pypos
Bi-gram HMM POS tagger written in Python

### Example usage

Python 3.

Please add `python` to your `%PATH%` / `$PATH` before executing the following commands:

    python build_tagger.py data/sents.train data/sents.devt model.out
    python run_tagger.py data/sents.test model.out sents.now.result
    
### Current accuracy on test set

~ 87.3% (55/63) strict match with `sents.out`

~ 92.1% (58/63) tolerating the tags for `Democrats`, `The` (White House), `six-month`

The remaining 5: `Labor`, `acceded`, `only`, `insisted`, `that` are not exactly tolerable. See `data/*.result` for more information.
