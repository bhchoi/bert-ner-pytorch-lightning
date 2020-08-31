from sklearn.model_selection import train_test_split


with open('data/train_data', mode='r', encoding='utf-8') as f:
    lines = f.readlines()

sentence_list = []
tag_list = []
words = []
tags = []
for line in lines:
    if len(line.rstrip('\n')) == 0:
        sentence_list.append(words)
        tag_list.append(tags)
        words = []
        tags = []
    else:  
        words.append(line.split('\t')[1])
        tags.append(line.split('\t')[2].rstrip('\n'))

print(sentence_list[0])

slot_labels = list(set([t for tags in tag_list for t in tags]))
slot_labels.sort()

with open('data/slot_labels.txt', mode='w', encoding='utf-8') as f:
    for label in slot_labels:
        f.write(label + '\n')

train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentence_list, tag_list, test_size=0.1, shuffle=True, random_state=1004)
train_sentences, val_sentences, train_tags, val_tags = train_test_split(train_sentences, train_tags, test_size=0.1)


def write_file(data_path, sentences, tags):
    with open(data_path, mode='w', encoding='utf-8') as f:
        for words, tags in zip(sentences, tags):
            for w, t in zip(words, tags):
                f.write('{}\t{}\n'.format(w, t))
            f.write('\n')

write_file('data/train_data.txt', train_sentences, train_tags)
write_file('data/val_data.txt', val_sentences, val_tags)
write_file('data/test_data.txt', test_sentences, test_tags)