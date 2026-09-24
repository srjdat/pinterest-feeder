import torch
from model import PinterestFeeder
from test_data import test_data
from transformers import CLIPProcessor, CLIPModel

# use mps if on mac, cuda if on nvidia gpu, else cpu
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = PinterestFeeder(1024).to(device)
model.load_state_dict(torch.load("models/pinterest_feeder1.pth", weights_only=True, map_location=device))
model.eval()

# path is automatically set to test folder so no need right here but you can change it if test folder is somewhere else
data = test_data(device=device)
queries = ['anime art', 'silver luxury car', 'fashion inspiration']
# print(data)

unique_images = list(set(item['image_filename'] for item in data))

validation_data = [
    item for item in data
    if item['image_filename'] in unique_images
]

embed_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device) # type: ignore
embed_model.eval()
process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

embedding_list = []

with torch.inference_mode():
    for image in data:

        for query in queries:
            text_info= process(text=query, return_tensors='pt', padding=True).to(device) # type: ignore
            text_features = embed_model.get_text_features(input_ids=text_info['input_ids'], attention_mask=text_info['attention_mask'])

            embedding_list.append(
                {
                    "embedding": torch.cat((image['image_embedding'], text_features.pooler_output), 1), # type: ignore
                    "query": query,
                    "image_filename": image['image_filename']
                }
            )

with torch.inference_mode():
    for embedding in embedding_list:
        score = model(embedding['embedding'][0])
        print(f"score: {torch.sigmoid(score)}, image: {embedding['image_filename']}, query: {embedding['query']}")
