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
        viterbi(tokens)

a, b = mat_transitions, map_emissions

def viterbi(T):
    global a # N*N: P(curr_tag|prev_tag) where N = num_tags
    global b # map: P(curr_tag|curr_token)

    mat_viterbi = [[0 for col in range(num_tags)] for row in range(T)]
    mat_backptr = [[-1 for col in range(num_tags)] for row in range(T)]

    for s in range(num_tags):
        mat_viterbi[s][0] = a[0][s] * b[T[0]][pos_tags_list[s]]
    for t in range(1, len(T)):
        for s in range(num_tags):
            v_max_prob, v_max_s, b_max_prob, b_max_s = 0, 0, 0, 0
            for s_prime in range(num_tags):
                b_curr_prob = mat_viterbi[s_prime][t-1] * a[s_prime][s]
                v_curr_prob = b_curr_prob * b[T[t-1]][pos_tags_list[s_prime]]
                if v_curr_prob > v_max_prob:
                    v_max_prob = v_curr_prob
                    v_max_s = s_prime
                if b_curr_prob > b_max_prob:
                    b_max_prob = b_curr_prob
                    b_max_s = s_prime
            mat_viterbi[s][t] = v_max_prob
            mat_backptr[s][t] = b_max_s

    for s_prime in range(num_tags):
        b_curr_prob = mat_viterbi[s_prime][-1] * a[s_prime][s]
        v_curr_prob = b_curr_prob * b[T[-1]][pos_tags_list[s_prime]]
        if v_curr_prob > v_max_prob:
            v_max_prob = v_curr_prob
            v_max_s = s_prime
        if b_curr_prob > b_max_prob:
            b_max_prob = b_curr_prob
            b_max_s = s_prime
    mat_viterbi[-1][-1] = v_max_prob
    mat_backptr[-1][-1] = b_max_s

    max_s = mat_backptr[-1][-1]
    fin_back_path = []
    for t in range(T-1, 0, -1):
        fin_back_path.append(max_s)
        max_s = mat_backptr[max_s][t-1]
    return fin_back_path[::-1]
