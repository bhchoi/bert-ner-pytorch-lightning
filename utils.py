
def load_slot_labels():
    return [label.rstrip('\n') for label in open('data/slot_labels.txt', mode='r', encoding='utf-8')]