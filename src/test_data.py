import torch
from image_loader import load_images
from transformers import CLIPProcessor, CLIPModel

# use mps if on mac, cuda if on nvidia gpu, else cpu
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# load images
# get image embeddings
# make a list of dicts with name, embedding, category (maybe)
@torch.inference_mode()
# i'm adding a device parameter, if the user specifically wants to use a cpu, mps, or cuda. but for my it's going to be mps since i'm in mac  
def test_data(path: str = "tests/", device=device):
    # path to images that we are going to embed
    # by default it's set to test/
    image_list = load_images(path=path)
    data = []

    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device) # type: ignore
    model.eval()
    process = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    for image in image_list:
        image_info = process(images=image, return_tensors='pt').to(device) # type: ignore
        image_features = model.get_image_features(pixel_values=image_info['pixel_values'])

        data.append(
            {
                "image_filename": image.filename,
                "image_embedding": image_features.pooler_output, # type: ignore
                "image": image
            }
        )

    # return the data that we're making
    return data

def main():
    test_data()

if __name__ == "__main__":
    main()
