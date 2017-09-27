from decimal import Decimal
import json
import sys

args = sys.argv

if len(args) != 4:
    sys.exit('Usage: python run_tagger.py test_file model_file result_file')

test_file, model_file, result_file = args[1:]
emit_file = 'emit.out'

mat_transitions, map_emissions = [], {}

start_tag, end_tag = '<s>', '</s>'

pos_tags_list = [start_tag]
pos_tags_dict = {start_tag: 0}
num_tags = 1

with open('pos.key', 'r') as k:
    for tag in k:
        stripped_tag = tag.strip()
        pos_tags_list.append(stripped_tag)
        pos_tags_dict[stripped_tag] = num_tags
        num_tags += 1
    pos_tags_list.append(end_tag)
    pos_tags_dict[end_tag] = num_tags
    num_tags += 1

with open(model_file, 'r') as m:
    num_tags = next(m)
    for line in m:
        mat_transitions.append(list(map(lambda p: Decimal(p), line.strip().split(' '))))

with open(emit_file, 'r') as e:
    map_emissions = json.load(e)

with open(test_file, 'r') as f, open(result_file, 'w') as g:
    for line in f:
        tokens = line.strip().split(' ')
        # viterbi(tokens, ...)
