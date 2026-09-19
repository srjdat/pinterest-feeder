import torch
from model import PinterestFeeder
from dataset import dataset

model = PinterestFeeder(1024)
model.load_state_dict(torch.load("models/pinterest_feeder1.pth", weights_only=True))
model.eval()
data = dataset(path='tests/', queries=['anime art', 'silver luxury car', 'fashion inspiration'])

unique_images = list(set(item['image_filename'] for item in data))

validation_data = [
    item for item in data
    if item['image_filename'] in unique_images
]

with torch.inference_mode():
    for item in validation_data:

        label = torch.tensor([item['label']], dtype=torch.float32) # get labels for each image item
        input_embedding = item['input'] # get input embeddings
        score = model(input_embedding) # get score form the model
        print(f"{torch.sigmoid(score)} image name: {item['image_filename']} query: {item['query']}")
