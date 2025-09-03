import numpy as np, open_clip, torch
from PIL import Image

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

original_vector = torch.from_numpy(feat_matrix[:2]).to(device) #First frame
images = [Image.open("./dataset/keyframes/L21_V001/00001.jpg"), 
          Image.open("./dataset/keyframes/L21_V001/00002.jpg")] #Obviously

with torch.no_grad():
    result = []
    #pretend like each is a batch
    for image in images:
        image_vector = model.encode_image(torch.stack([preprocess(image)]).to(device))
        image_vector /= torch.norm(image_vector, dim=-1, keepdim=True)
        result.append(image_vector)
    result = torch.concatenate(result, dim=0)

print(100 * (torch.einsum("bn,bn->b", result, original_vector) * 0.5 + 0.5))