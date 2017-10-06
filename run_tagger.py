from copy import deepcopy
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
    num_tags = int(next(m))
    for line in m:
        mat_transitions.append(list(map(lambda p: Decimal(p), line.strip().split(' '))))

with open(emit_file, 'r') as e:
    map_emissions = json.load(e)

a = mat_transitions

def viterbi(tokens, b):
    global a

    # a:
    #   N*N matrix where N = num_tags (includes <s> and </s>, already +2)
    #   P(curr_tag|prev_tag)
    #   a[prev_tag][curr_tag]

    # b:
    #   len(N) map where N = num_tags (includes <s> and </s>, already +2)
    #   P(curr_token|curr_tag)
    #   if b[curr_tag][curr_token] then b[curr_tag][curr_token] else 0

    T = len(tokens)

    mat_viterbi = [[0 for token in range(T)] for tag in range(num_tags)]
    mat_backptr = [[0 for token in range(T)] for tag in range(num_tags)]

    for s in range(num_tags):
        token, tag = tokens[0], pos_tags_list[s]
        if token in b[tag]:
            mat_viterbi[s][0] = a[0][s] * Decimal(b[tag][token])

    for t in range(1, T):
        for s in range(num_tags):
            v_max_prob, b_max_prob, b_max_s = 0, 0, 0
            for s_prime in range(num_tags):
                b_curr_prob = mat_viterbi[s_prime][t-1] * a[s_prime][s]
                v_curr_prob = 0
                token, tag = tokens[t], pos_tags_list[s]
                if token in b[tag]:
                    v_curr_prob = b_curr_prob * Decimal(b[tag][token])
                if v_curr_prob > v_max_prob:
                    v_max_prob = v_curr_prob
                if b_curr_prob > b_max_prob:
                    b_max_prob = b_curr_prob
                    b_max_s = s_prime
            mat_viterbi[s][t] = v_max_prob
            mat_backptr[s][t-1] = b_max_s

    v_max_prob, b_max_prob, b_max_s = 0, 0, 0
    for s_prime in range(num_tags):
        b_curr_prob = mat_viterbi[s_prime][-1] * a[s_prime][-1]
        v_curr_prob = 0
        token, tag = tokens[-1], pos_tags_list[s_prime]
        if token in b[tag]:
            v_curr_prob = b_curr_prob * Decimal(b[tag][token])
        if v_curr_prob > v_max_prob:
            v_max_prob = v_curr_prob
        if b_curr_prob > b_max_prob:
            b_max_prob = b_curr_prob
            b_max_s = s_prime
    mat_viterbi[-1][-1] = v_max_prob
    mat_backptr[-1][-1] = b_max_s

    max_s = -1
    fin_back_path = []
    for t in range(T-1, -1, -1):
        max_s = mat_backptr[max_s][t]
        fin_back_path.append(max_s)

    return fin_back_path[::-1]

def get_updated_emission_map(tokens, b):
    vocabulary = deepcopy(b);
    unseen = {tag: set() for tag in pos_tags_dict if tag != '<s>' and tag != '</s>'}
    for token in tokens:
        for tag in b:
            if tag == '<s>' or tag == '</s>':
                continue
            if token not in b[tag]:
                unseen[tag].add(token)
    for tag_index in range(1, num_tags-1):
        tag = pos_tags_list[tag_index]
        T = len(vocabulary[tag])
        Z = len(unseen[tag])
        C = len(b[tag])
        b[tag] = {token: ((b[tag][token] * C) / (C + T)) for token in b[tag]}
        for token in unseen[tag]:
            b[tag][token] = T / (Z * (C + T))
    return b

with open(test_file, 'r') as f, open(result_file, 'w') as g:
    for line in f:
        tokens = line.strip().split(' ')
        tags = list(map(lambda index: pos_tags_list[index], viterbi(tokens, get_updated_emission_map(tokens, map_emissions))))

        tagged_tokens = []
        for i in range(len(tokens)):
            tagged_tokens.append('{0}/{1}'.format(tokens[i], tags[i]))
        g.write('{0}\n'.format(' '.join(tagged_tokens)))
