import torch
import random
from torch.utils.data import DataLoader
from torchmultimodal.models.flava.model import flava_model_for_classification
from transformers import BertTokenizer
from transformers import AlbertTokenizer
from transformers import DistilBertTokenizer
from functools import partial
from typing import List
from torch import nn
from torchvision import transforms
from datasets import load_dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
MAX_LENGTH = 512

#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding="max_length", max_length=MAX_LENGTH)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", padding="max_length", max_length=MAX_LENGTH)
#tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", padding="max_length", max_length=MAX_LENGTH)

dataset = load_dataset('pubmed_qa', 'pqa_labeled')
print(dataset)

def preprocess(batch):
    label_map = {'yes': 1, 'no': 0, 'maybe': 2}  # mapping from string labels to integer labels
    labels = [label_map[i] for i in batch['final_decision']]
    
    tokenized = tokenizer(batch['question'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    tokenized['final_decision'] = labels

    return tokenized

# Preprocess the dataset
#train_dataset = dataset['train'].map(preprocess, batched=True)
test_dataset = dataset['train'].map(preprocess, batched=True)
print(test_dataset)
BATCH_SIZE = 1

#train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("test:", test_loader)
print("Preprocess completed")

#Define the FLAVA model for the ranking task
num_classes = 3  # single output neuron for relevance score
model = flava_model_for_classification(num_classes=num_classes)
model = model.to(device)

# Set the learning rate and optimizer
learning_rate = 1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
"""
# Define the pairwise ranking loss
class PairwiseRankingLoss(nn.MarginRankingLoss):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(margin=margin)

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #target = torch.ones_like(pos_scores)
        return super().forward(pos_scores, neg_scores, target)

# loss_fn = PairwiseRankingLoss()
# """
dummy_image = torch.zeros((BATCH_SIZE, 3, 224, 224)).to(device)

def process_batch(batch):
    images, labels, questions = [], [], []
    for i, label in enumerate(batch["final_decision"]):
        
        question = batch["input_ids"][i]
        questions.append(question)
        images.append(dummy_image[i])
        labels.append(label)
    
    rquestions = torch.stack(questions).to(device)
    rlabels = torch.tensor(labels, dtype=torch.long).to(device)
    #im = np.asarray(images.resize((224,224)))
    rimages = torch.stack(images).to(device)
    
    return rquestions, rimages, rlabels
# def process_batch(batch):
#     images, labels, questions = [], [], []
#     for i, label in enumerate(batch["final_decision"]):
        
#         question = batch["input_ids"][i]
#         questions.append(question)
        
#         image = batch["image"][i]
#         # Resize the image to the same size as dummy_image
#         transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#         image = transform(image)
#         images.append(image)
        
#         labels.append(label)
    
#     rquestions = torch.stack(questions).to(device)
#     rlabels = torch.tensor(labels, dtype=torch.long).to(device)
#     rimages = torch.stack(images).to(device)
    
#     return rquestions, rimages, rlabels

"""
# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        
        # Get the positive and negative pairs from the batch
        pos_questions, pos_images, pos_labels, neg_questions, neg_images, neg_labels = process_batch(batch)
        # Forward pass
        optimizer.zero_grad()
        if pos_questions is not None:
            print(pos_questions)
        if pos_questions is not None: 
            
            pos_logits = model(text=pos_questions, image=pos_images, labels=pos_labels).logits
            neg_logits = model(text=neg_questions, image=neg_images, labels=neg_labels).logits

            min_size = min(pos_logits.size(0), neg_logits.size(0))
            target = torch.ones(min_size).to(device)
            ranking_loss = loss_fn(pos_logits[:min_size, 1], neg_logits[:min_size, 1], target)
            print(f"Batch {i+1} - pos_logits: {pos_logits}")
            print(f"Batch {i+1} - neg_logits: {neg_logits}")
            print(f"Batch {i+1} - pos_logits[:, 1]: {pos_logits[:, 1]}")
            print(f"Batch {i+1} - neg_logits[:, 1]: {neg_logits[:, 1]}")
            print(f"Batch {i+1} - ranking_loss: {ranking_loss.item()}")
            # Backward pass
            ranking_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += ranking_loss.item()
            #print("Batch {} and Loss {}".format(batchnum,ranking_loss))
         
         # Get the questions, images, and labels from the batch
        questions, images, labels = process_batch2(batch)
        #print("labels: ", labels)
        #print("images: ", images)
        #print("questions: ", questions)
        # Forward pass
        optimizer.zero_grad()

        outputs  = model(text=questions, image=images, labels=labels)

        #loss = loss_fn(logits, labels)
        #print(outputs.logits)
        loss = outputs.loss
        print(f"Batch {i+1} - loss: {loss}")
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
"""
# Testing loop
model.eval()

# Initialize variables to calculate the performance metric (e.g., accuracy)
correct_predictions = 0
total_predictions = 0

# Iterate over the test dataset
for i, batch in enumerate(test_loader):
    # Get the questions, images, and labels from the batch
    questions, images, labels = process_batch(batch)

    # Forward pass (no gradient calculation is needed)
    with torch.no_grad():
        outputs = model(text=questions, image=images, labels=labels)

    # Get the predicted labels
    #_, predicted_labels = torch.max(outputs.logits, 1)
    #print(outputs.logits.size())
    predicted_labels = torch.argmax(outputs.logits, dim= 1)

    #print("predictions : ", predicted_labels)
    # Calculate the performance metric (e.g., accuracy)
    correct_predictions += (predicted_labels == labels).sum().item()
    total_predictions += labels.size(0)

# Calculate and print the final performance metric
accuracy = correct_predictions / total_predictions
#print(f"Flava's BertTokenizer Accuracy: {accuracy * 100:.2f}%")
#print(f"Flava's AlbertTokenizer Accuracy: {accuracy * 100:.2f}%")
print(f"Flava's DistilBertTokenizer Accuracy: {accuracy * 100:.2f}%")
