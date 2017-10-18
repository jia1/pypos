export PATH=$PATH:/usr/local/Python-3.6.3/bin
nohup time python3 cross_validate.py data/sents.train &> nohup-cross-validate-05-2.out &
