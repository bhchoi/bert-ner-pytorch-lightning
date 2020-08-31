from preprocessor import Preprocessor
from torch.utils.data import Dataset
from utils import load_slot_labels

class NerDataset(Dataset):
    def __init__(self, data_path: str, preprocessor: Preprocessor):
        self.sentence_list = []
        self.tag_list = []
        self.preprocessor = preprocessor
        self.slot_labels = load_slot_labels()

        self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        words = []
        tags = []
        for line in lines:
            if len(line.rstrip('\n')) == 0:
                self.sentence_list.append(words)
                self.tag_list.append(tags)
                words = []
                tags = []
            else:  
                words.append(line.split('\t')[0])
                tags.append(line.split('\t')[1].rstrip('\n'))

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        sentence = self.sentence_list[idx]
        tags = self.tag_list[idx]
        tags = [self.slot_labels.index(t) if t in self.slot_labels else self.preprocessor.tokenizer.unk_token for t in tags]

        input_ids, slot_labels, attention_mask, token_type_ids = self.preprocessor.get_input_features(sentence, tags)

        if len(slot_labels) != 64:
            print(sentence)
            print(len(input_ids))
            print(len(slot_labels))
            print(len(attention_mask))
            print(len(token_type_ids))

        return input_ids, slot_labels, attention_mask, token_type_ids
    