import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

from src.model import BertForSST2
from src.dataset import SST2_Dataset
from src.configs import SST2_Config


class Trainer_SST2:
    def __init__(self, config=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(self, batch):
        return ([s.text_a for s in batch], [s.label for s in batch])

    def test(self, model, dataloader, tokenizer):
        model.eval()
        softmax = nn.Softmax(dim=1)
        predict_labels = []
        test_labels = []
        for batch in dataloader:
            inputs = batch[0]
            labels = batch[1]
            inputs = tokenizer(inputs, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            logits = model(**inputs).cpu()
            logits = softmax(logits)
            predict_label = torch.argmax(logits, dim=1)
            predict_labels += [i.item() for i in predict_label]
            test_labels += labels

        # acc
        return accuracy_score(predict_labels, test_labels)

    def train(self, config=None):
        config = SST2_Config()
        max_epochs = config.max_epochs
        test_steps = config.test_steps
        accu_steps = config.accu_steps
        data_dir = config.data_dir
        batch_size = config.batch_size
        train_dataloader = DataLoader(SST2_Dataset(
            data_dir, "train"), batch_size=batch_size, collate_fn=self.collate_fn, shuffle=True)
        test_dataloader = DataLoader(SST2_Dataset(
            data_dir, "test"), batch_size=batch_size, collate_fn=self.collate_fn)
        dev_dataloader = DataLoader(SST2_Dataset(
            data_dir, "dev"), batch_size=batch_size, collate_fn=self.collate_fn)

        # model
        model = BertForSST2(config).to(self.device)
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        ce = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=config.lr)
        total_steps = int(len(train_dataloader)/accu_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

        # train
        model.train()
        for epoch in range(max_epochs):
            all_loss = None
            for i, batch in enumerate(train_dataloader):
                # print(batch)
                inputs = batch[0]
                labels = torch.LongTensor(batch[1]).to(self.device)
                inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                logits = model(**inputs)
                loss = ce(logits, labels)
                all_loss = loss if all_loss is None else loss+all_loss
                if i % accu_steps == 0:
                    print("step", i, "loss", all_loss)
                    all_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    all_loss=None
                if i % test_steps == 0:
                    acc = self.test(model, test_dataloader,tokenizer)
                    model.train()
                    print(f"step:{i} acc:{acc}")
