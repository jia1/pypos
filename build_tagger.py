import json
import sys

args = sys.argv

if len(args) != 4:
    sys.exit('Usage: python build_tagger.py train_file devt_file model_file')

train_file, devt_file, model_file = args[1:]

mat_transitions, map_emissions = [], {}
num_transitions, num_emissions = {}, {}
sum_transitions, sum_emissions = {}, {}

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

with open(train_file, 'r') as f:
    for line in f:
        prev_token, prev_tag = '', start_tag
        split_line = line.strip().split(' ')
        for tagged_token in split_line:
            split_token = tagged_token.split('/')
            curr_token = ''.join(split_token[:-1])
            curr_tag = split_token[-1]

            if prev_tag not in num_transitions:
                num_transitions[prev_tag] = {curr_tag: 1}
                sum_transitions[prev_tag] = 1
            elif curr_tag not in num_transitions[prev_tag]:
                num_transitions[prev_tag][curr_tag] = 1
                sum_transitions[prev_tag] += 1
            else:
                num_transitions[prev_tag][curr_tag] += 1
                sum_transitions[prev_tag] += 1

            if curr_token not in num_emissions:
                num_emissions[curr_token] = {curr_tag: 1}
                sum_emissions[curr_token] = 1
            elif curr_tag not in num_emissions[curr_token]:
                num_emissions[curr_token][curr_tag] = 1
                sum_emissions[curr_token] += 1
            else:
                num_emissions[curr_token][curr_tag] += 1
                sum_emissions[curr_token] += 1

            prev_token, prev_tag = curr_token, curr_tag

        curr_tag = '</s>'
        if prev_tag not in num_transitions:
            num_transitions[prev_tag] = {curr_tag: 1}
            sum_transitions[prev_tag] = 1
        elif curr_tag not in num_transitions[prev_tag]:
            num_transitions[prev_tag][curr_tag] = 1
            sum_transitions[prev_tag] += 1
        else:
            num_transitions[prev_tag][curr_tag] += 1
            sum_transitions[prev_tag] += 1

mat_transitions = [[0 for col in range(num_tags)] for row in range(num_tags)]

for prev_tag in num_transitions:
    row = pos_tags_dict[prev_tag]
    for next_tag in num_transitions[prev_tag]:
        col = pos_tags_dict[next_tag]
        mat_transitions[row][col] = num_transitions[prev_tag][next_tag] / sum_transitions[prev_tag]

for curr_token, curr_emission in num_emissions.items():
    map_emissions[curr_token] = {k: (v / sum_emissions[curr_token]) for k, v in curr_emission.items()}

'''
with open(devt_file, 'r') as g:
    for line in g:
        prev_token, prev_tag = '', start_tag
        split_line = line.strip().split(' ')
        for tagged_token in split_line:
            curr_token, curr_tag = tagged_token.split('/')

            prev_token, prev_tag = curr_token, curr_tag
'''
with open(model_file, 'w') as h:
    h.write('{0}\n'.format(num_tags))
    for row in mat_transitions:
        h.write('{0}\n'.format(' '.join([str(col) for col in row])))

with open('emit.out', 'w') as e:
    json.dump(map_emissions, e)
