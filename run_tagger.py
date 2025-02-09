from __future__ import division

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

# 1. Same as (1) in build_tagger, so as to ease the reading of the transition matrix (2D list)
with open('pos.key', 'r') as k:
    for tag in k:
        stripped_tag = tag.strip()
        pos_tags_list.append(stripped_tag)
        pos_tags_dict[stripped_tag] = num_tags
        num_tags += 1
    pos_tags_list.append(end_tag)
    pos_tags_dict[end_tag] = num_tags
    num_tags += 1

# 2. Load transition matrix into memory
with open(model_file, 'r') as m:
    num_tags = int(next(m))
    for line in m:
        mat_transitions.append(list(map(lambda p: Decimal(p), line.strip().split(' '))))

# 3. Load emission map into memory
with open(emit_file, 'r') as e:
    map_emissions = json.load(e)

a = mat_transitions

# 4. Viterbi function which accepts a list of tokens and a smoothed emission map and returns a list of POS tags (by index)
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

    # 4a. Initialize viterbi matrix with <s> -> tag values from transition matrix
    for s in range(num_tags):
        token, tag = tokens[0], pos_tags_list[s]
        if token in b[tag]:
            mat_viterbi[s][0] = a[0][s] * Decimal(b[tag][token])

    # 4b. Iteratively populate the viterbi matrix
    for t in range(1, T):
        for s in range(num_tags):
            v_max_prob, b_max_prob, b_max_s = 0, 0, 0
            # 4bi. Get argmax s' (POS tag) based on the viterbi probabilities in the previous t (token)
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

    # 4c. Populate final column of viterbi matrix with tag -> </s> values from transition matrix
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

    # 4d. Walk through the backpointer matrix to get a path of POS tags (in reverse)
    max_s = -1
    fin_back_path = []
    for t in range(T-1, -1, -1):
        max_s = mat_backptr[max_s][t]
        fin_back_path.append(max_s)

    # 4e. Return the path of POS tags in sequence from first to last token
    return fin_back_path[::-1]

# 5. Function to smooth the emission map with a somewhat biased Witten Bell smoothing
def get_updated_emission_map(tokens, b):
    unseen = []
    # 5a. Add unseen test tokens to the unseen list
    for token in tokens:
        seen = False
        for tag in b:
            if token in b[tag]:
                seen = True
                break
        if not seen:
            unseen.append(token)
    # 5b. Smooth if there are unseen test tokens
    if unseen:
        num_unseen = len(unseen)
        vocabulary = set(tokens)
        for tag in b:
            vocabulary.update(b[tag].keys())
        Z = num_unseen * num_tags
        T = len(vocabulary) - Z
        for tag in b:
            tag_freq = len(b[tag])
            b[tag] = {token: ((b[tag][token] * tag_freq) / (tag_freq + T)) for token in b[tag]}
            for unseen_token in unseen:
                b[tag][unseen_token] = T / (Z * (tag_freq + T))
    return b

# 6. For each test case (line), re-smooth the original emission map and pass it to the viterbi function
with open(test_file, 'r') as f, open(result_file, 'w') as g:
    for line in f:
        tokens = line.strip().split(' ')
        tags = list(map(lambda index: pos_tags_list[index], viterbi(tokens, get_updated_emission_map(tokens, map_emissions))))

        # 7. Write the tags to file
        tagged_tokens = []
        for i in range(len(tokens)):
            tagged_tokens.append('{0}/{1}'.format(tokens[i], tags[i]))
        g.write('{0}\n'.format(' '.join(tagged_tokens)))
