import sys

args = sys.argv

if len(args) != 4:
    sys.exit('Usage: python build_tagger.py train_file devt_file model_file')

train_file, devt_file, model_file = args[1:]

mat_transitions, map_emissions = [], []
num_transitions, num_emissions = {}, {}

pos_tags_list = []
pos_tags_dict = {}
num_tags = 0

sum_key = 'TOTAL'
start_tag = '<s>'

with open('pos.key') as k:
    for tag in k:
        pos_tags_list.append(tag)
        pos_tags_dict[tag] = num_tags
        num_tags += 1

with open(train_file) as f:
    prev_token, prev_tag = '', start_tag
    for line in f:
        split_line = line.split(' ')
        for tagged_token in split_line:
            curr_token, curr_tag = tagged_token.split('/')

            if prev_tag not in num_transitions:
                num_transitions[prev_tag] = {curr_tag: 1, sum_key: 1}
            elif curr_tag not in num_transitions[prev_tag]:
                num_transitions[prev_tag][curr_tag] = 1
                num_transitions[prev_tag][sum_key] += 1
            else:
                num_transitions[prev_tag][curr_tag] += 1
                num_transitions[prev_tag][sum_key] += 1

            if curr_token not in num_emissions:
                num_emissions[curr_token] = {curr_tag: 1, sum_key: 1}
            elif curr_tag not in num_emissions[curr_token]:
                num_emissions[curr_token][curr_tag] = 1
                num_emissions[curr_token][sum_key] += 1
            else:
                num_emissions[curr_token][curr_tag] += 1
                num_emissions[curr_token][sum_key] += 1

            prev_token, prev_tag = curr_token, curr_tag

mat_transitions = [[0 for col in range(num_tags)] for row in range(num_tags)]

for prev_tag in num_transitions:
    row = pos_tags_dict[prev_tag]
    for next_tag in num_transitions[prev_tag]:
        col = pos_tags_dict[next_tag]
        mat_transitions[row][col] = \
            num_transitions[prev_tag][next_tag] / num_transitions[prev_tag][sum_key]

map_emissions = num_emissions

'''
with open(devt_file) as g:
    prev_token, prev_tag = '', start_tag
    for line in g:
        split_line = line.split(' ')
        for tagged_token in split_line:
            curr_token, curr_tag = tagged_token.split('/')

            prev_token, prev_tag = curr_token, curr_tag
'''
'''
with open(model_file) as h:
    h.write(num_tags)
    for row in mat_transitions:
        pass
'''
