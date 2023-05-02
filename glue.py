# Step 1: Import the necessary libraries
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
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding="max_length", max_length=MAX_LENGTH)
#tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", padding="max_length", max_length=MAX_LENGTH)
#tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", padding="max_length", max_length=MAX_LENGTH)

# Step 2: Load the GLUE dataset (e.g., MRPC)
dataset = load_dataset("glue", "mrpc")
print(dataset)
# Step 3: Preprocess the dataset using the DistilBertTokenizer
def preprocess(batch):
    combined_sentences = [f"{sent1} [SEP] {sent2}" for sent1, sent2 in zip(batch['sentence1'], batch['sentence2'])]
    tokenized = tokenizer(combined_sentences, padding='max_length', truncation=True, max_length=MAX_LENGTH)
    return tokenized

#train_dataset = dataset['train'].map(preprocess, batched=True)
test_dataset = dataset['test'].map(preprocess, batched=True)
BATCH_SIZE = 32
#print(train_dataset[10:])
# Step 4: Create DataLoaders for the train and test datasets
#train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Step 5: Define the FLAVA model for the ranking task
num_classes = 2  # single output neuron for relevance score
model = flava_model_for_classification(num_classes=num_classes)
model = model.to(device)
# Step 6: Set the learning rate and optimizer
#learning_rate = 1e-7
optimizer = torch.optim.Adam(model.parameters())
# Step 7: Create a function to process the batch
dummy_image = torch.zeros((BATCH_SIZE, 3, 224, 224)).to(device)

def process_batch(batch):
    images, labels, questions = [], [], []
    for i, label in enumerate(batch["label"]):
        #print(i)
        question = batch["input_ids"][i]
        #print(question)
        #print(questions)
        questions.append(question)
        images.append(dummy_image[i])
        labels.append(label)
    
    rquestions = torch.stack(questions).to(device)
    rlabels = torch.tensor(labels, dtype=torch.long).to(device)
    #im = np.asarray(images.resize((224,224)))
    rimages = torch.stack(images).to(device)
    return rquestions, rimages, rlabels
    
model.eval()
acc_lst = []
for _ in range(5):
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
        _, predicted_labels = torch.max(outputs.logits, 1)

        # Calculate the performance metric (e.g., accuracy)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate and print the final performance metric
    #print("One run done!")
    accuracy = correct_predictions / total_predictions
    print("Accuracy: ", accuracy)
    acc_lst.append(accuracy)
meann = sum(acc_lst)/len(acc_lst)
# Step 8: Test the model and calculate the accuracy
print(f"Flava's BertTokenizer Accuracy on GLUE MRPC dataset: {meann * 100:.2f}%")
#print(f"Flava's AlbertTokenizer Accuracy on GLUE MRPC dataset: {meann * 100:.2f}%")
#print(f"Flava's DistilBertTokenizer Accuracy on GLUE MRPC dataset: {meann * 100:.2f}%")
