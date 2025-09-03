import open_clip, \
       torch, \
       numpy as np, \
       os, \
       matplotlib.pyplot as plt, \
       matplotlib.image as mpimg, \
       pandas as pd

from tqdm import tqdm

model_name = "EVA02-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, 
    device=device, 
    pretrained="merged2b_s4b_b131k")
model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(model_name)

text = "dog"

all_videos = []

for video in tqdm(sorted(os.scandir("./dataset/clip_features"), key=lambda f: f.name), desc="Loading CLIP features", position=0):
    if video.is_file():
        frame_vectors = np.load(video.path)
        video_tag = os.path.splitext(video.name)[0]
        for frame_idx, vector in enumerate(frame_vectors):
            all_videos.append([video_tag, frame_idx, vector])

all_videos = pd.DataFrame(all_videos, columns=["video", "frame", "vector"])

all_frames = torch.from_numpy(np.stack(all_videos["vector"].to_numpy())).to(device)

#Calculate embedding vector
with torch.no_grad():
    text_vector = model.encode_text(tokenizer(text).to(device))
    text_vector /= torch.norm(text_vector, dim=-1, keepdim=True)
    
#Simple normalization trick from [-1, 1] of cos() to [0, 1] of probability
values, indices = torch.topk((((all_frames @ text_vector.T) * 0.5 + 0.5) * 100).flatten(), k=10)
values = values.cpu().numpy()
indices = indices.cpu().numpy()

#Show the best result
cols = 5
rows = (len(indices) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
fig.suptitle(f"Query: {text}")

for i, ax in tqdm(enumerate(axes.flat), desc="Composing output"):
    if i < len(indices):
        video_tag, frame_idx = all_videos[["video", "frame"]].iloc[indices[i]]
        img = mpimg.imread(f"./dataset/keyframes/{video_tag}/{frame_idx:05d}.jpg")

        ax.imshow(img)
        ax.set_title(f"{video_tag}/{frame_idx:05d}: {values[i]:.2f}%", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()