import os
import json
from string import Template

print("Loading PyTorch...", end=' ', flush=True)
import numpy as np
import torch
import torch.nn.functional as F
print("DONE")

from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response
import uvicorn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def from_numpy(tensor):
    return torch.from_numpy(tensor).to(device)

print("Loading open-clip model...", end=' ', flush=True)
import open_clip

# Load the model using open-clip
model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr", device=device)
tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
print("DONE")

CLIP_FEATURES_PATH = "dataset/clipv3"
videos = [video[:-4] for video in os.listdir(CLIP_FEATURES_PATH)]

print("Loading dataset...", end=' ', flush=True)
videos_features = []
timestamps = []
frame_indices = []
watch_urls = []

fps = {}

for video in videos:
    videos_features.append(from_numpy(np.load(f"{CLIP_FEATURES_PATH}/{video}.npy")))

    with open(f"dataset/map-keyframes/{video}.csv") as map_keyframes:
        pts_time = []
        frame_idx = []

        next(map_keyframes)
        for rows in map_keyframes:
            cells = [cell.strip() for cell in rows.split(',')]
            pts_time.append(float(cells[1]))
            frame_idx.append(int(cells[3]))
            if video in fps:
                assert fps[video] == float(cells[2])
            else:
                fps[video] = float(cells[2])

        timestamps.append(pts_time)
        frame_indices.append(frame_idx)

    with open(f"dataset/metadata/{video}.json", encoding="utf-8") as metadata:
        payload = json.load(metadata)
        watch_urls.append(payload["watch_url"])

print("DONE")


zero = torch.zeros(1).to(device)
max_len: int = max(feature.shape[0] for feature in videos_features)

mask_full = torch.triu(torch.ones((max_len, max_len)).to(device))

all_videos = []
all_indices = []

for video_idx, video_features in enumerate(videos_features):
    all_videos.append(torch.full((video_features.shape[0],), video_idx).to(device))
    all_indices.append(torch.arange(0, video_features.shape[0]).to(device))

all_videos = torch.cat(all_videos)
all_indices = torch.cat(all_indices)

def search_all_queries(queries, k):
    with torch.no_grad():
        # Tokenize and encode text queries
        text_features = [model.encode_text(tokenizer(query).to(device)) for query in queries]

        # Normalize text features
        text_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in text_features]

        all_values = []

        for video_features in videos_features:
            # Normalize video features
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            text_features_view = [feature.expand(video_features.shape[0], -1) for feature in text_features]
            similarities = [F.cosine_similarity(view, video_features) * 0.5 + 0.5 for view in text_features_view]
            
            mask = mask_full[1:video_features.shape[0], 1:video_features.shape[0]]

            count = len(queries)
            score = similarities[count - 1]
            for i in range(count - 1, 0, -1):
                score_mat = score[1:].view(1, -1).expand(video_features.shape[0] - 1, -1)
                score = torch.cat([(mask * score_mat).max(1)[0], zero]) + similarities[i - 1]

            # Apply softmax to get probabilities
            all_values.append(F.softmax(score * 100 / count, dim=-1))

        final_values, final_indices = torch.cat(all_values).topk(k, sorted=True)
        final_videos = all_videos[final_indices]
        final_frames = all_indices[final_indices]

    return final_videos, final_frames, final_values


with open("web/template.html", encoding="utf-8") as html_file:
    mapping = dict(fps=json.dumps(fps))
    template = Template(html_file.read())

    with open("web/style.css", encoding="utf-8") as css_file:
        mapping["css"] = css_file.read()
    with open("web/script.js", encoding="utf-8") as js_file:
        mapping["js"] = js_file.read()

    with open("config.json", encoding="utf-8") as config_file:
        mapping["config"] = json.dumps(json.loads(config_file.read()))

    html = template.substitute(mapping)

async def homepage(_: Request):
    return Response(html, media_type="text/html")


async def search(request: Request):
    request_queries = (await request.body()).decode()
    queries = [s.strip() for s in request_queries.split(">>")]

    html = ""
    result = search_all_queries(queries, 1000)

    for vid, idx, val in zip(*result):
        video = videos[vid]
        image_url = f"/keyframes/{video}/{idx.item() + 1:03d}.jpg"
        timestamp = timestamps[vid][idx]
        watch_url = f"{watch_urls[vid]}&t={round(timestamp)}"
        score = val

        html += f"""
        <div class="result-item" data-video-id="{video}" data-time-id="{int(timestamp*1000)}" href="{watch_url}" onclick="fill(this)">
            <a href="{watch_url}" target="_blank" rel="noopener noreferrer" onclick="visit(this)">
                <img class="result-thumbnail" src="{image_url}" loading="lazy" />
            </a>

            <div class="result-stats">
                <span>{video}:{int(timestamp*1000)}</span>
                <span>{score:.2f}%</span>
            </div>
        </div>
        """

    return Response(html, media_type="text/html")


app = Starlette(routes=[
    Route("/", homepage, methods=["GET"]),
    Route("/search", search, methods=["POST"]),
    Mount("/keyframes", StaticFiles(directory="downscaled"), name="keyframes")
])

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=3333)
