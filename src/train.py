from transformers.models.oneformer.modeling_oneformer import PredictionBlock

import early_stopping
from model import PinterestFeeder
import torch
import torch.nn as nn
from dataset import dataset
import random
import torch.optim.adamw
from early_stopping import EarlyStopping

# use mps if on mac, cuda if on nvidia gpu, else cpu
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# make the model
model = PinterestFeeder(1024).to(device)
early_stopping = EarlyStopping(patience=5, delta=0.001) # if we start overfitting, stop early

# hyperparameters
learning_rate = 1e-4
weight_decay = 1e-2

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# scoring
criteration = nn.BCEWithLogitsLoss()

# get the data from dataset.py
# has features: query: str, image_filename, label, combined_embedding
data = dataset(train=True, path="data/")
num_epochs = 100

# split for trianing and validation is 80/20
# create a way to implement randomized training and validation images per epoch
# create split for this epoch
# create list of images WITHOUT any duplicates (dataset has each image 3 times because of queries)
unique_images = sorted({item['image_filename'] for item in data})
random.Random(42).shuffle(unique_images)

# don't randomize right now, just have the same split every time to see what happens
# random.shuffle(unique_images) # shuffle it before splitting to get it random


# split the images based on predefined training/validation split (80/20)
training_images = unique_images[:(int(len(unique_images) * .8))]
validation_images= unique_images[(int(len(unique_images) * .8)):]
#
# create training/validation data list so we can iterate through it in the loops
training_data = [
    item for item in data
    if item['image_filename'] in training_images
]
validation_data = [
    item for item in data
    if item['image_filename'] in validation_images
]

# print(training_data)

# checking if a specific image is in training or validation
# where = "train" if any(item['image_filename'] == "/Users/srjdat/Development/pinterest-feeder/data/fashion_adidas_jacket_2.JPG" for item in training_data) else "not test"
# print(where)
# where = "validation" if any(item['image_filename'] == "/Users/srjdat/Development/pinterest-feeder/data/fashion_adidas_jacket_2.JPG" for item in validation_data) else "not validation"
# print(where)


# training and validation loop
for epoch in range(num_epochs):

    # calculate loss
    train_loss = 0
    validation_loss = 0

    # training
    # create randomized list of training_images
    random.shuffle(training_data)

    model.train()
    # training loop
    for item in training_data:
        # starts with fresh gradients each time
        optimizer.zero_grad()

        # get label and input embeddings
        label = torch.tensor([item['label']], dtype=torch.float32, device=device)
        input_embedding = item['input'].to(device)

        # score using our model
        score = model(input_embedding[0]) # get score form the model

        # calculate loss and optimize
        loss = criteration(score, label) # calculate loss
        loss.backward(retain_graph=True) # backward propagation
        optimizer.step() # optimize the weights

        # add to total training loss
        train_loss += loss

    # validation loop
    model.eval()
    mistakes = []
    with torch.inference_mode():
        for item in validation_data:
            label = torch.tensor([item['label']], dtype=torch.float32, device=device)
            input_embedding = item['input'].to(device)
            score = model(input_embedding[0]) # get score form the model
            loss = criteration(score, label) # calculate loss

            # add to total validation loss
            validation_loss += loss

    # print losses each epoch
    print(f"epoch {epoch}: training loss: {train_loss / len(training_data)}, validation loss: {validation_loss / len(validation_data)}")
    # print(f"epoch {epoch}: training loss: {train_loss / len(training_data)}")

    early_stopping((validation_loss/len(validation_data)).item(), model) # type: ignore
    if early_stopping.early_stop:
        print("Early Stopping due to overfitting")
        break

early_stopping.load_best_model(model)

# validation mistakes on the best
mistake = []
model.eval()
with torch.inference_mode():
    for item in validation_data:
        logit = model(item['input'].to(device)[0])
        probability = torch.sigmoid(logit).item()
        prediction = int(probability >= .5)
        actual = int(item['label'])

        if prediction != actual:
            mistake.append(
                {
                    "image": item["image_filename"],
                    "query": item["query"],
                    "label": actual,
                    "probability": probability,
                    "error": abs(probability - actual),
                }
            )

print(mistake)

torch.save(model.state_dict(), "models/pinterest_feeder1.pth")
