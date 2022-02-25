from datasets import load_metric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model import BertForSST2
from dataset import SST2_Dataset
from configs import SST2_Config

class Trainer_SST2:
    def __init__(self, config):
        self.config = config
    
    def collate_fn(batch):
        return ([s.text_a for s in batch], [s.label for s in batch])
    
    def test(model,dataloader,tokenizer):
        model.eval()
        softmax = nn.Softmax(dim=1)
        metric = load_metric("accuracy")
        predcit_labels = []
        test_labels = []
        for batch in dataloader:
            inputs = batch[0]
            labels = batch[1]
            inputs = tokenizer(inputs,padding=True,truncation=True,max_length=512,return_tensors="pt")
            logits = model(**inputs).cpu()
            logits = softmax(logits)
            predict_label = torch.argmax(logits,dim=1)
            predcit_labels += predict_label
            test_labels += labels

        # acc
        return metric.compute(predictions=predict_label,references=test_labels)

    
    def train(self,config):
        config = SST2_Config()
        max_epochs = config.max_epochs
        test_steps = config.test_steps
        accu_steps = config.accu_steps
        data_dir = config.data_dir
        batch_size = config.batch_size
        train_dataloader = DataLoader(SST2_Dataset(data_dir,"train"),batch_size=batch_size,collate_fn=self.collate_fn,shuffle=True)
        test_dataloader = DataLoader(SST2_Dataset(data_dir,"test"),batch_size=batch_size,collate_fn=self.collate_fn)
        dev_dataloader = DataLoader(SST2_Dataset(data_dir,"dev"),batch_size=batch_size,collate_fn=self.collate_fn)

        # model
        model = BertForSST2(config)
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        ce = nn.CrossEntropyLoss
        optimizer = AdamW(model.parameters(), lr=config.lr)
        total_steps = int(len(train_dataloader)/accu_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0.1*total_steps,num_training_steps=total_steps)
        

        # train
        model.train()
        for epoch in max_epochs:
            loss = 0.0
            for i, batch in enumerate(train_dataloader):
                inputs = batch[0]
                labels = batch[1]
                inputs = tokenizer(inputs,padding=True,truncation=True,max_length=512,return_tensors="pt")
                logits = model(**inputs)
                loss += ce(logits,labels)
                if i%accu_steps == 0:
                    print("step", i,"loss",loss)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if i%test_steps == 0:
                    acc = self.test(model,test_dataloader)
                    print(f"step:{i} acc:{acc}")













