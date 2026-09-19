from model import PinterestFeeder
import torch
import torch.nn as nn
from dataset import dataset
import random
import torch.optim.adamw

# make the model
model = PinterestFeeder(1024)

# hyperparameters
learning_rate = 1e-4
weight_decay = 1e-2

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# scoring
criteration = nn.BCEWithLogitsLoss()

# get the data from dataset.py
# has features: query: str, image_filename, label, combined_embedding
data = dataset()
num_epochs = 10

# split for trianing and validation is 80/20
# create a way to implement randomized training and validation images per epoch
# create split for this epoch
# create list of images WITHOUT any duplicates (dataset has each image 3 times because of queries)
unique_images = list(set(item['image_filename'] for item in data))
random.shuffle(unique_images) # shuffle it before splitting to get it random

# split the images based on predefined training/validation split (80/20)
training_images = unique_images[:(int(len(unique_images) * .8))]
validation_images= unique_images[(int(len(unique_images) * .8)):]

# create training/validation data list so we can iterate through it in the loops
training_data = [
    item for item in data
    if item['image_filename'] in training_images
]
validation_data = [
    item for item in data
    if item['image_filename'] in validation_images
]

# training and validation loop
for epoch in range(num_epochs):

    # calculate loss
    train_loss = 0
    validation_loss = 0

    # training
    # create randomized list of training_images
    random.shuffle(training_data)

    # training loop
    for item in training_data:

        # starts with fresh gradients each time
        optimizer.zero_grad()

        # get label and input embeddings
        label = torch.tensor([item['label']], dtype=torch.float32)
        input_embedding = item['input']

        # score using our model
        score = model(input_embedding[0]) # get score form the model

        # calculate loss and optimize
        loss = criteration(score, label) # calculate loss
        loss.backward(retain_graph=True) # backward propagation
        optimizer.step() # optimize the weights

        # add to total training loss
        train_loss += loss

    # validation loop
    with torch.inference_mode():
        for item in validation_data:

            label = torch.tensor([item['label']], dtype=torch.float32) # get labels for each image item
            input_embedding = item['input'] # get input embeddings
            score = model(input_embedding[0]) # get score form the model
            loss = criteration(score, label) # calculate loss

            # add to total validation loss
            validation_loss += loss

    # print losses each epoch
    print(f"training loss: {train_loss / len(training_data)}, validation loss: {validation_loss / len(validation_data)}")

torch.save(model.state_dict(), "models/pinterest_feeder1.pth")
