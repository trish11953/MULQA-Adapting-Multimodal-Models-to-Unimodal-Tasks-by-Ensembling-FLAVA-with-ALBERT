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
from datasets import load_dataset
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding="max_length", max_length=MAX_LENGTH)
# tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", padding="max_length", max_length=MAX_LENGTH)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", padding="max_length", max_length=MAX_LENGTH)

# Load the dataset
dataset = load_dataset("commonsense_qa")
# Make a new label column from 'answer_key' column
label_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}
#labels = [label_map[label] for label in batch['answerKey']]
# Define the preprocess function
def preprocess(batch):
    tokenized = tokenizer(batch['question'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    labels = [label_map[label] for label in batch['answerKey']]
    tokenized['answer_label'] = labels
    return tokenized

# Preprocess the dataset
train_dataset = dataset['train'].map(preprocess, batched=True)
test_dataset = dataset['validation'].map(preprocess, batched=True)

# print(train_dataset[1])

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
#print("Preprocess completed")

# Define the FLAVA model for the ranking task
num_classes = 5  # single output neuron for relevance score
model = flava_model_for_classification(num_classes=num_classes)
model = model.to(device)

# Set the learning rate and optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dummy_image = torch.zeros((BATCH_SIZE, 3, 224, 224)).to(device)

def process_batch(batch):
    images, labels, questions = [], [], []
    for i, label in enumerate(batch["answer_label"]):
        #print(i)
        question = batch["input_ids"][i]
        #print(question)
        #print(questions)
        questions.append(question)
        images.append(dummy_image[i])
        labels.append(label)
    #print(labels)
    rquestions = torch.stack(questions).to(device)
    rlabels = torch.tensor(labels, dtype=torch.long).to(device)
    #im = np.asarray(images.resize((224,224)))
    rimages = torch.stack(images).to(device)
    return rquestions, rimages, rlabels

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
    predicted_labels = torch.argmax(outputs.logits, dim=1)

    # Calculate the performance metric (e.g., accuracy)
    correct_predictions += (predicted_labels == labels).sum().item()
    total_predictions += labels.size(0)

# Calculate and print the final performance metric
accuracy = correct_predictions / total_predictions
# print(f"Flava's BertTokenizer Accuracy: {accuracy * 100:.2f}%")
# print(f"Flava's AlbertTokenizer Accuracy: {accuracy * 100:.2f}%")
print(f"Flava's DistilBertTokenizer Accuracy: {accuracy * 100:.2f}%")


"""
# Define the training function
def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in dataloader:
        questions, images, labels = process_batch(batch)

        optimizer.zero_grad()
        outputs = model(text=questions, image=images, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_labels = torch.max(outputs.logits, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)
    print("In Training")
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy

# (Optional) Define the validation function
def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            questions, images, labels = process_batch(batch)
            outputs = model(text=questions, image=images, labels=labels)

            running_loss += outputs.loss.item()
            _, predicted_labels = torch.max(outputs.logits, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
    print("In validation")
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy

# Set the number of epochs
num_epochs = 3

# Main training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, device)
    # (Optional) Perform validation
    val_loss, val_accuracy = validate(model, test_loader, device)
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

"""