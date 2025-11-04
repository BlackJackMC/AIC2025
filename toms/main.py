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

print("Loading CLIP model...", end=' ', flush=True)
import clip
model, preprocess = clip.load("ViT-B/32", device=device)
print("DONE")

CLIP_FEATURES_PATH = "dataset/clip-features-32"
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
        text_features = [model.encode_text(clip.tokenize(query).to(device)).view(1, -1) for query in queries]

        all_values = []

        for video_features in videos_features:
            text_features_view = [feature.expand(video_features.shape[0], -1) for feature in text_features]
            similarities = [F.cosine_similarity(view, video_features) * 0.5 + 0.5 for view in text_features_view]

            mask = mask_full[1:video_features.shape[0], 1:video_features.shape[0]]

            count = len(queries)
            score = similarities[count - 1]
            for i in range(count - 1, 0, -1):
                score_mat = score[1:].view(1, -1).expand(video_features.shape[0] - 1, -1)
                score = torch.cat([(mask * score_mat).max(1)[0], zero]) + similarities[i - 1]

            all_values.append(score * 100 / count)

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
        frame_index = frame_indices[vid][idx]
        score = val

        html += f"""
        <div class="result-item" data-video-id="{video}" data-time-id="{frame_index}" href="{watch_url}" onclick="fill(this)">
            <a href="/video/{video}?frame={frame_index}"rel="noopener noreferrer" onclick="visit(this)">

                <img class="result-thumbnail" src="{image_url}" />
            </a>


            <div class="result-stats">
                <span>{video}:{frame_index}</span>
                <span>{score:.2f}%</span>
            </div>
        </div>
        """

    return Response(html, media_type="text/html")

from string import Template

async def video_page(request: Request):
    video = request.path_params["video"]
    idxs = frame_indices[videos.index(video)]

    # Build frame HTML
    frames_html = ""
    for i, idx in enumerate(idxs):
        num = i + 1
        image_url = f"/keyframes/{video}/{num:03d}.jpg"
        frames_html += f"""
        <div class="frame" id="frame-{idx}">
          <img src="{image_url}" loading="lazy" />
          <div>{video}:{idx}</div>
        </div>
        """

    # Load template
    with open("web/video.html", encoding="utf-8") as f:
        template = Template(f.read())

    html = template.substitute(video=video, frames=frames_html)
    return Response(html, media_type="text/html")



app = Starlette(routes=[
    Route("/", homepage, methods=["GET"]),
    Route("/search", search, methods=["POST"]),
    Route("/video/{video}", video_page, methods=["GET"]),
    Mount("/keyframes", StaticFiles(directory="keyframes"), name="keyframes"),
    Mount("/web", StaticFiles(directory="web"), name="web"),
])

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=1111)
