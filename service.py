import os

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from concurrent.futures import ThreadPoolExecutor

@dataclass(frozen=True)
class Clip4clip:
    name: str = "Searchium-ai/clip4clip-webvid150k"
    
@dataclass(frozen=True)
class Dataset:
    base: Path = Path("./dataset")

    @property
    def videos(self) -> Path:
        return self.base / "segments"

    @property
    def videos_features(self) -> Path:
        return self.base / "clip-features"

    @property
    def transcript_features(self) -> Path:
        return self.base / "transcript-features"
    

CLIP_data = Clip4clip()
dataset = Dataset()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained(CLIP_data.name)
text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_data.name).to(device)


def parallel_load_features(base_path, video_names, desc, device, num_workers=8):  
    def load_file(video_name):
        file_path = f"{base_path}/{video_name}.npy"
        return torch.from_numpy(np.load(file_path))

    results = thread_map(
        load_file, 
        video_names, 
        max_workers=num_workers, 
        desc=desc
    )
    
    return torch.stack(results).to(device)

def fetch_text_feature(text: str | list[str]):
    inputs = tokenizer(text=text, return_tensors="pt", padding=True).to(device)
    outputs = text_model(**inputs)

    outputs = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
    return outputs

@torch.no_grad()
def search_all_queries(queries, k):
    text_features = fetch_text_feature(queries).to(device) #(queries, 512)
    print("Encoded queries")
    #(videos, queries)
    queries_video_sim = torch.einsum("ab,cb->ac", video_features, text_features) * 0.5 + 0.5
    print("Calculated similarity scores")
    #(videos, queries)
    # queries_transcript_sim = torch.einsum("ab,cb->ac", transcript_features, text_features) * 0.5 + 0.5
    
    dp = torch.zeros(video_features.shape[0], text_features.shape[0]) #dp[v][q]
    V, Q = dp.shape
    
    dp[:, Q - 1] = queries_video_sim[:, Q - 1]
    for i in tqdm(range(Q - 2, -1, -1), desc="Calculating toms score"):
        max_future_score = 0.0
        
        for t in tqdm(range(V - 2, -1, -1), leave=False):
            max_future_score = max(max_future_score, dp[t + 1, i + 1])
            dp[t, i] = queries_video_sim[t, i] + max_future_score
            
        dp[V - 1, i] = queries_video_sim[V - 1, i]
        
    result = dp[:, 0]
    result = (result * 100) / Q
    selected = torch.topk(result, k).indices.squeeze(0)
    final_values = result[selected]
    final_videos = video_names[selected]
        
    print("Done")
    if k == 1:
        return [final_videos], [final_values]
    return list(final_videos), final_values.tolist()

WORKERS = os.cpu_count() or 8
print(f"Total workers: {WORKERS}")

#Load all video names to a list
print("Loading videos")
video_names = np.array([os.path.splitext(f)[0] for f in tqdm(os.listdir(dataset.videos), desc="Videos") if os.path.splitext(f)[1] == '.mp4'])
print(f"Total videos: {video_names.shape}")

# Load video features
print("Loading video features")
video_features = parallel_load_features(
    dataset.videos_features, 
    video_names, 
    "Video features", 
    device, 
    num_workers=WORKERS
)
print(f"Total video features: {video_features.shape}")

# Load transcript features 
# print("Loading transcript features")
# transcript_features = parallel_load_features(
#     dataset.transcript_features, 
#     video_names, 
#     "Transcript features", 
#     device, 
#     num_workers=WORKERS
# )
# print(f"Total transcript features: {transcript_features.shape}")
