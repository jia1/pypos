from copy import deepcopy
from decimal import Decimal
import json
import random
import sys

args = sys.argv

if len(args) != 4:
    sys.exit('Usage: python build_tagger.py train_file devt_file model_file')

train_file, devt_file, model_file = args[1:]

start_tag, end_tag = '<s>', '</s>'

pos_tags_list = [start_tag]
pos_tags_dict = {start_tag: 0}
num_tags = 1

mat_transitions, map_emissions = [], {}
num_transitions, num_emissions = {}, {}
sum_transitions, sum_emissions = {}, {}

with open('pos.key', 'r') as k:
    for tag in k:
        stripped_tag = tag.strip()
        pos_tags_list.append(stripped_tag)
        pos_tags_dict[stripped_tag] = num_tags
        num_tags += 1
    pos_tags_list.append(end_tag)
    pos_tags_dict[end_tag] = num_tags
    num_tags += 1

mat_transitions = [[0 for tag in range(num_tags)] for tag in range(num_tags)]
tag_count_dict = {tag: 0 for tag in pos_tags_list}
tag_token_dict = {tag: {} for tag in pos_tags_list}
tag_tag_dict = {tag: {tag: 0 for tag in pos_tags_list} for tag in tag_count_dict}
num_transitions = deepcopy(tag_tag_dict)
map_emissions = deepcopy(tag_token_dict)
num_emissions = deepcopy(tag_token_dict)
sum_transitions = deepcopy(tag_count_dict)
sum_emissions = deepcopy(tag_count_dict)

lines = []
with open(train_file, 'r') as f:
    for line in f:
        lines.append(line)
        prev_token, prev_tag = '', start_tag
        split_line = line.strip().split(' ')
        for tagged_token in split_line:
            split_token = tagged_token.split('/')
            curr_token = ''.join(split_token[:-1])
            curr_tag = split_token[-1]

            num_transitions[prev_tag][curr_tag] += 1
            sum_transitions[prev_tag] += 1

            if curr_token not in num_emissions[curr_tag]:
                num_emissions[curr_tag][curr_token] = 1
                sum_emissions[curr_tag] += 1
            else:
                num_emissions[curr_tag][curr_token] += 1
                sum_emissions[curr_tag] += 1

            prev_token, prev_tag = curr_token, curr_tag

        curr_tag = '</s>'
        num_transitions[prev_tag][curr_tag] += 1
        sum_transitions[prev_tag] += 1

with open(devt_file, 'r') as f:
    for line in f:
        lines.append(line)
        prev_token, prev_tag = '', start_tag
        split_line = line.strip().split(' ')
        for tagged_token in split_line:
            split_token = tagged_token.split('/')
            curr_token = ''.join(split_token[:-1])
            curr_tag = split_token[-1]

            num_transitions[prev_tag][curr_tag] += 1
            sum_transitions[prev_tag] += 1

            if curr_token not in num_emissions[curr_tag]:
                num_emissions[curr_tag][curr_token] = 1
                sum_emissions[curr_tag] += 1
            else:
                num_emissions[curr_tag][curr_token] += 1
                sum_emissions[curr_tag] += 1

            prev_token, prev_tag = curr_token, curr_tag

        curr_tag = '</s>'
        num_transitions[prev_tag][curr_tag] += 1
        sum_transitions[prev_tag] += 1

for prev_tag in num_transitions:
    count = sum_transitions[prev_tag]
    if count > 0:
        row = pos_tags_dict[prev_tag]
        for next_tag in num_transitions[prev_tag]:
            col = pos_tags_dict[next_tag]
            mat_transitions[row][col] = num_transitions[prev_tag][next_tag] / sum_transitions[prev_tag]

for curr_tag, curr_emission in num_emissions.items():
    map_emissions[curr_tag] = {k: (v / sum_emissions[curr_tag]) for k, v in curr_emission.items()}

with open(model_file, 'w') as h:
    h.write('{0}\n'.format(num_tags))
    for row in mat_transitions:
        h.write('{0}\n'.format(' '.join([str(col) for col in row])))

with open('emit.out', 'w') as e:
    json.dump(map_emissions, e)

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
            mat_viterbi[s][0] = Decimal(a[0][s]) * Decimal(b[tag][token])

    for t in range(1, T):
        for s in range(num_tags):
            v_max_prob, b_max_prob, b_max_s = 0, 0, 0
            for s_prime in range(num_tags):
                b_curr_prob = mat_viterbi[s_prime][t-1] * Decimal(a[s_prime][s])
                v_curr_prob = 0
                token, tag = tokens[t], pos_tags_list[s]
                if token in b[tag]:
                    v_curr_prob = Decimal(b_curr_prob) * Decimal(b[tag][token])
                if v_curr_prob > v_max_prob:
                    v_max_prob = v_curr_prob
                if b_curr_prob > b_max_prob:
                    b_max_prob = b_curr_prob
                    b_max_s = s_prime
            mat_viterbi[s][t] = v_max_prob
            mat_backptr[s][t-1] = b_max_s

    v_max_prob, b_max_prob, b_max_s = 0, 0, 0
    for s_prime in range(num_tags):
        b_curr_prob = mat_viterbi[s_prime][-1] * Decimal(a[s_prime][-1])
        v_curr_prob = 0
        token, tag = tokens[-1], pos_tags_list[s_prime]
        if token in b[tag]:
            v_curr_prob = Decimal(b_curr_prob) * Decimal(b[tag][token])
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
    unseen = []
    for token in tokens:
        seen = False
        for tag in b:
            if token in b[tag]:
                seen = True
                break
        if not seen:
            unseen.append(token)
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

num_transitions = deepcopy(tag_tag_dict)
map_emissions = deepcopy(tag_token_dict)
num_emissions = deepcopy(tag_token_dict)
sum_transitions = deepcopy(tag_count_dict)
sum_emissions = deepcopy(tag_count_dict)

k = 10
cut = len(lines) // 10
score = 0
shuffled_lines = deepcopy(lines)
for t in range(k):
    random.shuffle(shuffled_lines)
    for line in lines[cut:]:
        prev_token, prev_tag = '', start_tag
        split_line = line.strip().split(' ')
        for tagged_token in split_line:
            split_token = tagged_token.split('/')
            curr_token = ''.join(split_token[:-1])
            curr_tag = split_token[-1]

            num_transitions[prev_tag][curr_tag] += 1
            sum_transitions[prev_tag] += 1

            if curr_token not in num_emissions[curr_tag]:
                num_emissions[curr_tag][curr_token] = 1
                sum_emissions[curr_tag] += 1
            else:
                num_emissions[curr_tag][curr_token] += 1
                sum_emissions[curr_tag] += 1

            prev_token, prev_tag = curr_token, curr_tag

        curr_tag = '</s>'
        num_transitions[prev_tag][curr_tag] += 1
        sum_transitions[prev_tag] += 1

    for prev_tag in num_transitions:
        count = sum_transitions[prev_tag]
        if count > 0:
            row = pos_tags_dict[prev_tag]
            for next_tag in num_transitions[prev_tag]:
                col = pos_tags_dict[next_tag]
                mat_transitions[row][col] = num_transitions[prev_tag][next_tag] / sum_transitions[prev_tag]

    a = mat_transitions

    for curr_tag, curr_emission in num_emissions.items():
        map_emissions[curr_tag] = {k: (v / sum_emissions[curr_tag]) for k, v in curr_emission.items()}

    for line in lines[:cut]:
        split_line = line.strip().split(' ')
        tokens, correct_tags = [], []
        for tagged_token in split_line:
            split_token = tagged_token.split('/')
            curr_token = ''.join(split_token[:-1])
            curr_tag = split_token[-1]
            tokens.append(curr_token)
            correct_tags.append(curr_tag)
        answer_tags = list(map(lambda index: pos_tags_list[index], viterbi(tokens, get_updated_emission_map(tokens, map_emissions))))
        for i in range(len(tokens)):
            if answer_tags[i] == correct_tags[i]:
                score += 1

print(score, k * cut)
