# pypos
Bi-gram HMM POS tagger written in Python

### Example usage

Python 3 please.

Please add `python` to your `%PATH%` / `$PATH` before executing the following commands:

    python build_tagger.py data/sents.train data/sents.devt model.out
    python run_tagger.py data/sents.test model.out sents.out
    
### Current accuracy

~ 87.3%
