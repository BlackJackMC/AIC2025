import os

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import numpy as np
import torch


MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"
DATASET_PATH = "./dataset"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(device)



#Load all video names to a list
video_names = np.array([os.path.splitext(f)[0] for f in os.listdir(f"{DATASET_PATH}/segments") if os.path.splitext(f)[1] == '.mp4'])
print(f"Total videos: {video_names.shape}")



#Load all video vectors to a tensor
video_features = torch.stack(
    [torch.from_numpy(np.load(f"{DATASET_PATH}/clip-features/{video}.npy")) for video in video_names]
).to(device) #(number of videos, 512)



def fetch_text_feature(text: str | list[str]):
    inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = text_model(**inputs)

    outputs = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
    return outputs

@torch.no_grad()
def search_all_queries(queries, k):
    text_features = fetch_text_feature(queries).to(device) #(queries, 512)
    similarity_score = torch.einsum("ab,cb->ac", video_features, text_features) #(videos, queries)
    similarity_score = similarity_score * 0.5 + 0.5
    dp = torch.zeros(video_features.shape[0], text_features.shape[0]) #dp[v][q]
    V, Q = dp.shape
    
    dp[:, Q - 1] = similarity_score[:, Q - 1]
    for i in range(Q - 2, -1, -1):
        max_future_score = 0.0
        
        for t in range(V - 2, -1, -1):
            max_future_score = max(max_future_score, dp[t + 1, i + 1])
            dp[t, i] = similarity_score[t, i] + max_future_score
            
        dp[V - 1, i] = similarity_score[V - 1, i]
        
    result = dp[:, 0]
    result = (result * 100) / Q
    selected = torch.topk(result, k).indices.squeeze(0)
    final_values = result[selected]
    final_videos = video_names[selected]
        

    return list(final_videos), final_values.tolist()
