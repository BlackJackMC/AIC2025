import numpy as np, open_clip, torch, os
from PIL import Image
from tqdm import tqdm

model_name = "EVA02-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, 
    device=device, 
    pretrained="merged2b_s4b_b131k")
model.to(device)
model.eval()

feat_matrix = np.load("./dataset/clip_features/L21_V001.npy")
print(feat_matrix.shape)

original_vector = torch.from_numpy(feat_matrix[:1]).to(device) 

# Comparing the embedding vectors with the corresponding frame yield no result
# That means the vectors and the frames order is mismatch
# images = [Image.open("./dataset/keyframes/L21_V001/00001.jpg")] #Obviously

# Test with all the dataset to make sure the error was order displacement by sorting the path
images_path = sorted(os.scandir("./dataset/keyframes/L21_V001"), key=lambda e: e.name)
with torch.no_grad():
    result = []
    batch_size = 64
    # pretend like each is a batch
    # for image in images:
    #     image_vector = model.encode_image(torch.stack([preprocess(image)]).to(device))
    #     image_vector /= torch.norm(image_vector, dim=-1, keepdim=True)
    #     result.append(image_vector)
    # real batch
    for i in tqdm(range(0, len(images_path), batch_size)):
        image_vector = model.encode_image(torch.stack([preprocess(Image.open(image)) for image in images_path[i:i+batch_size]]).to(device))
        image_vector /= torch.norm(image_vector, dim=-1, keepdim=True)
        result.append(image_vector)
    result = torch.concatenate(result, dim=0)

print(torch.topk((100 * (torch.einsum("an,bn->ab", result, original_vector) * 0.5 + 0.5)).flatten(), k=1))