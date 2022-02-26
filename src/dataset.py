import os
from torch.utils.data import Dataset
from transformers.data.processors import InputExample,InputFeatures

class SST2_Dataset(Dataset):
    def _create_examples(self, file_dir, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        f = open(file_dir,"r")
        lines = f.readlines()
        f.close()
        for (i, line) in enumerate(lines):
            if i == 0 and set_type in ["train","dev"]:
                continue
            if set_type == "test":
                line = line.strip().split(" ")
                guid = "%s-%s" % (set_type, i)
                text_a = " ".join(line[1:])
                label = int(line[0])
            else:
                line = line.strip().split("\t")
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = int(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def __init__(self,data_dir,set_type="train") -> None:
        file_dir = os.path.join(data_dir,f"{set_type}.tsv")
        self.dataset = self._create_examples(file_dir, set_type)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
        