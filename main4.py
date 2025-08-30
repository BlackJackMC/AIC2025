import os, uvicorn, json, numpy as np, torch

from open_clip import create_model_and_transforms
from llm2vec import LLM2Vec
from transformers import AutoModel, AutoTokenizer, AutoConfig

from string import Template
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response

print("Loading OpenCLIP model...", end=' ', flush=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, _, preprocess = create_model_and_transforms(
    model_name='EVA02-L-14',
    pretrained='merged2b_s4b_b131k',
    device=device
)
model.eval()

llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(
    llm_model_name, trust_remote_code=True
)
llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

CLIP_FEATURES_PATH = "dataset/clip-features"
videos = [video[:-4] for video in os.listdir(CLIP_FEATURES_PATH)]

print("Loading dataset...", end=' ', flush=True)
videos_features = []
frame_indices = []
timestamps = []
watch_urls = []

for video in videos:
    video_features = torch.from_numpy(np.load(f"{CLIP_FEATURES_PATH}/{video}.npy")).to(device)
    num_frames = video_features.shape[0]
    frame_idx = [i for i in range(num_frames) if i % 3 == 0]
    if not frame_idx:
        continue
    filtered_features = video_features[frame_idx]
    videos_features.append(filtered_features)
    frame_indices.append(frame_idx)
    pts_time = [i * 10 / 30 for i in frame_idx]
    timestamps.append(pts_time)

    with open(f"dataset/media-info/{video}.json", encoding="utf-8") as metadata:
        payload = json.load(metadata)
        watch_urls.append(payload["watch_url"])

print(f"DONE (Loaded {len(videos_features)} videos with {sum(v.shape[0] for v in videos_features)} total frames)")

print("Preparing tensors...", end=' ', flush=True)
zero = torch.zeros(1).to(device)
max_len = max(feature.shape[0] for feature in videos_features) if videos_features else 1
mask_full = torch.triu(torch.ones((max_len, max_len)).to(device))

all_videos = []
all_indices = []

for video_idx, video_features in enumerate(videos_features):
    all_videos.append(torch.full((video_features.shape[0],), video_idx).to(device))
    all_indices.append(torch.arange(0, video_features.shape[0]).to(device))

all_videos = torch.cat(all_videos) if all_videos else torch.tensor([], dtype=torch.long, device=device)
all_indices = torch.cat(all_indices) if all_indices else torch.tensor([], dtype=torch.long, device=device)
print("DONE")

def search_all_queries(queries, k):
    print("Processing search queries...", end=' ', flush=True)
    if not videos_features or not queries:
        print("DONE")
        return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.float, device=device)

    with torch.no_grad():
        text_features = [model.encode_text(l2v.encode(query, convert_to_tensors=True).to(device)) for query in queries]
        text_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in text_features]

        all_values = []

        for video_features in videos_features:
            video_features_norm = video_features / video_features.norm(dim=-1, keepdim=True)
            text_features_view = [feature.expand(video_features.shape[0], -1) for feature in text_features]
            similarities = [(view @ video_features_norm) * 0.5 + 0.5 for view in text_features_view]

            count = len(queries)
            score = similarities[count - 1]

            if score.dim() == 0:
                score = score.unsqueeze(0)

            if video_features.shape[0] > 1:
                mask = mask_full[1:video_features.shape[0], 1:video_features.shape[0]]
                for i in range(count - 1, 0, -1):
                    score_mat = score[1:].view(1, -1).expand(video_features.shape[0] - 1, -1)
                    max_scores = (mask * score_mat).max(1)[0]
                    score = torch.cat([max_scores, zero]) + similarities[i - 1]
                    if score.dim() == 0:
                        score = score.unsqueeze(0)
            else:
                score = similarities[0]

            all_values.append(score * 100 / count)

        total_frames = sum(v.shape[0] for v in all_values)
        k = min(k, total_frames) if total_frames > 0 else 0

        if k == 0:
            print("DONE")
            return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.float, device=device)

        final_values, final_indices = torch.cat(all_values).topk(k, sorted=True)
        final_videos = all_videos[final_indices]
        final_frames = all_indices[final_indices]

    print("DONE")
    return final_videos, final_frames, final_values

print("Loading web templates...", end=' ', flush=True)
with open("web/template.html", encoding="utf-8") as html_file:
    mapping = dict(fps=json.dumps({}))
    template = Template(html_file.read())

    with open("web/style.css", encoding="utf-8") as css_file:
        mapping["css"] = css_file.read()
    with open("web/script.js", encoding="utf-8") as js_file:
        mapping["js"] = js_file.read()

    with open("config.json", encoding="utf-8") as config_file:
        mapping["config"] = json.dumps(json.loads(config_file.read()))

    html = template.substitute(mapping)
print("DONE")

async def homepage(_: Request):
    print("Serving homepage...", end=' ', flush=True)
    response = Response(html, media_type="text/html")
    print("DONE")
    return response

async def search(request: Request):
    print("Handling search request...", end=' ', flush=True)
    request_queries = (await request.body()).decode()
    queries = [s.strip() for s in request_queries.split(">>")]
    print(f"Search query: {queries}")

    html = ""
    result = search_all_queries(queries, 1000)

    for vid, idx, val in zip(*result):
        video = videos[vid]
        original_index = frame_indices[vid][idx]
        frame_index = original_index * 10
        timestamp = timestamps[vid][idx]
        image_url = f"/keyframes/{video}/{original_index + 1:05d}.jpg"
        watch_url = f"{watch_urls[vid]}&t={round(timestamp)}"
        score = val

        html += f"""
        <div class="result-item" data-video-id="{video}" data-time-id="{frame_index}" href="{watch_url}" onclick="fill(this)">
            <a href="{watch_url}" target="_blank" rel="noopener noreferrer" onclick="visit(this)">
                <img class="result-thumbnail" src="{image_url}" loading="lazy" />
            </a>

            <div class="result-stats">
                <span>{video}:{frame_index}</span>
                <span>{score:.2f}%</span>
            </div>
        </div>
        """

    response = Response(html, media_type="text/html")
    print("DONE")
    return response

print("Initializing web server...", end=' ', flush=True)
app = Starlette(routes=[
    Route("/", homepage, methods=["GET"]),
    Route("/search", search, methods=["POST"]),
    Mount("/keyframes", StaticFiles(directory="downscaled/keyframe"), name="keyframes")
])
print("DONE")

if __name__ == "__main__":
    print("Starting server...", end=' ', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=1111)
    print("DONE")