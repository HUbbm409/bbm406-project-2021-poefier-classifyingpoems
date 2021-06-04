import sys

if len(sys.argv) != 5:
    print("Not enough arguments.")
    print(sys.argv[1:])
    exit()
if not(sys.argv[2] == "age" or sys.argv[2] == "type"):
    print("Prediction can either be \'age\' or \'type\'. You entered: {}".format(sys.argv[2]))
    exit()
try:
    float(sys.argv[3])
except:
    print("Learning rate can only be float. You entered: {}".format(sys.argv[3]))
    exit()
if not sys.argv[4].isdigit():
    print("Epoch number can only be integer. You entered: {}".format(sys.argv[4]))
    exit()

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#

import pandas as pd

df = pd.read_csv(sys.argv[1])
df.dropna()

predict_what = sys.argv[2]
number_of_labels = 0
if predict_what == "age":
    df["age"] = df["age"].replace(["Renaissance", "Modern"], [0, 1])
    number_of_labels = 2
    X = df['content']
    y = df['age']
else:
    df["type"] = df["type"].replace(["Mythology & Folklore", "Nature", "Love"], [0, 1, 2])
    number_of_labels = 3
    X = df['content']
    y = df['type']

#

from sklearn.model_selection import train_test_split

train_text, test_text, train_labels, test_labels = train_test_split(X, y, test_size=0.30)

train_labels = train_labels.tolist()
train_text = train_text.tolist()
test_labels = test_labels.tolist()
test_text = test_text.tolist()

#

assert len(train_labels) == len(train_text)
assert len(test_labels) == len(test_text)

#

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#

print("Original: ", train_text[0])
print("Length: ", len(train_text[0]))
print("\n")

print("Tokenized: ", tokenizer.tokenize(train_text[0]))
print("Length: ", len(tokenizer.tokenize(train_text[0])))
print("\n")

print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_text[0])))
print("Length: ", len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_text[0]))))
print("\n")


#

def text_to_id(tokenizer, text_list):
    ids_list = []
    for item in text_list:
        encoded_item = tokenizer.encode(item, add_special_tokens=True)
        ids_list.append(encoded_item)
    return ids_list


#

train_text_ids = text_to_id(tokenizer, train_text)
test_text_ids = text_to_id(tokenizer, test_text)

#

print('Original: {}\n'.format(train_text[0]))
print('Token IDs: {}\n'.format(train_text_ids[0]))
print("len(train_text_ids) = {}\n".format(len(train_text_ids)))
print("len(test_text_ids) = {}".format(len(test_text_ids)))

#

print('Train: max sentence length: ', max([len(sen) for sen in train_text_ids]))
print('Train: Min sentence length: ', min([len(sen) for sen in train_text_ids]))
print('Test: max sentence length: ', max([len(sen) for sen in test_text_ids]))
print('Test: Min sentence length: ', min([len(sen) for sen in test_text_ids]))


#

def padding_truncating(input_ids_list, max_length):
    processed_input_ids_list = []
    for item in input_ids_list:
        seq_list = []
        if len(item) < max_length:
            # Define a seq_list with the length of max_length
            seq_list = [0] * (max_length - len(item))
            item = item + seq_list
        elif len(item) >= max_length:
            item = item[:max_length]
        processed_input_ids_list.append(item)
    return processed_input_ids_list


#

train_padding_list = padding_truncating(train_text_ids, max_length=50)
test_padding_list = padding_truncating(test_text_ids, max_length=50)


#

def get_attention_masks(pad_input_ids_list):
    attention_masks_list = []
    for item in pad_input_ids_list:
        mask_list = []
        for subitem in item:
            if subitem > 0:
                mask_list.append(1)
            else:
                mask_list.append(0)
        attention_masks_list.append(mask_list)
    return attention_masks_list


#

train_attention_masks = get_attention_masks(train_padding_list)
test_attention_masks = get_attention_masks(test_padding_list)

#

assert len(train_text) == len(train_labels) == len(train_attention_masks) == len(train_padding_list)
assert len(test_text) == len(test_labels) == len(test_attention_masks) == len(test_padding_list)

#

train_padding_list, validation_padding_list, train_labels, validation_labels, train_attention_masks, validation_attention_masks = train_test_split(
    train_padding_list, train_labels, train_attention_masks, random_state=2020, test_size=0.1)

#

assert len(train_labels) == len(train_attention_masks) == len(train_padding_list)
assert len(validation_labels) == len(validation_attention_masks) == len(validation_padding_list)
assert len(test_labels) == len(test_attention_masks) == len(test_padding_list)

#

print("len(train_labels) = {}\nlen(validation_labels) = {}\nlen(test_labels) = {}".format(len(train_labels),
                                                                                          len(validation_labels),
                                                                                          len(test_labels)))

#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

train_inputs = torch.tensor(train_padding_list)
validation_inputs = torch.tensor(validation_padding_list)
test_inputs = torch.tensor(test_padding_list)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_labels)

train_masks = torch.tensor(train_attention_masks)
validation_masks = torch.tensor(validation_attention_masks)
test_masks = torch.tensor(test_attention_masks)

#

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

#

from transformers import BertForSequenceClassification, AdamW, BertConfig
import torch

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=number_of_labels,
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

#

epsilon = 1e-8


learning_rate = float(sys.argv[3])
epoch_count = int(sys.argv[4])


#

optimizer = AdamW(model.parameters(),
                  lr=learning_rate,
                  eps=epsilon
                  )

#

from transformers import get_linear_schedule_with_warmup

epochs = epoch_count

total_steps = len(train_dataloader) * epochs
print("total_steps = {}".format(total_steps))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

#

import numpy as np


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#

import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


import random

seed_val = 12345

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

loss_values = []
accuracies_list = []

for epoch_i in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    accuracies_list.append(eval_accuracy / nb_eval_steps)
