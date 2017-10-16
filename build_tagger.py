from __future__ import division

from copy import deepcopy
import json
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

# 1. Create data structures to map from index to POS tag (list) and from POS tag to index (dict)
# This is to make sense of each index in the 2D transition matrix
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

# 2. Open the training data file and parse the tokens and tags
# Count tag -> tag for transition matrix and tag -> word for emission map
with open(train_file, 'r') as f:
    for line in f:
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

# 3. Open the dev data file and do the same as (2)
with open(devt_file, 'r') as f:
    for line in f:
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

# 4. Translate transition counts (on a dictionary) to transition matrix (2D list)
# Divide the frequency of tagA -> tagB by the total frequency of tagA
for prev_tag in num_transitions:
    count = sum_transitions[prev_tag]
    if count > 0:
        row = pos_tags_dict[prev_tag]
        for next_tag in num_transitions[prev_tag]:
            col = pos_tags_dict[next_tag]
            mat_transitions[row][col] = num_transitions[prev_tag][next_tag] / sum_transitions[prev_tag]

# 5. Iterate through the emission map and divide the frequency of tag -> word by tag
for curr_tag, curr_emission in num_emissions.items():
    map_emissions[curr_tag] = {k: (v / sum_emissions[curr_tag]) for k, v in curr_emission.items()}

# 6. Write transition matrix to file
with open(model_file, 'w') as h:
    h.write('{0}\n'.format(num_tags))
    for row in mat_transitions:
        h.write('{0}\n'.format(' '.join([str(col) for col in row])))

# 7. Write emission map to file
with open('emit.out', 'w') as e:
    json.dump(map_emissions, e)
