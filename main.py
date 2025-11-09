import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from service import search_all_queries, search_all_transcript

app = FastAPI()

app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

@app.get('/segments/{segment}')
async def get_segment(segment: str):
    return FileResponse(f"./dataset/segments/{segment}.mp4", 
                        media_type="video/mp4", 
                        headers= {
                            "Accept-Ranges": "bytes"
                        })

@app.get('/videos/{video}')
async def get_video(video: str):
    return FileResponse(f"./dataset/videos_480p/{video}.mp4", 
                        media_type="video/mp4", 
                        headers= {
                            "Accept-Ranges": "bytes"
                        })
    
class SearchResponse(BaseModel):
    videos: list[str]
    accuracy: list[float]

@app.get('/transcript', response_model=SearchResponse)
def search_transcript(queries: str, k: int = 1):
    print(f"/transcript queries=\"{queries}\" k={k}")
    
    videos, accuracy = search_all_transcript(queries, k)

    # Convert NumPy types and flatten nested lists
    videos = [str(v) for v in videos]
    accuracy = [
        float(a[0]) if isinstance(a, (list, tuple)) else float(a)
        for a in accuracy
    ]

    print(f"/transcript videos={videos} accuracy={accuracy}")
    return {"videos": videos, "accuracy": accuracy}



@app.get('/videos', response_model=SearchResponse)
def search(queries: str, k: int = 1):
    print(f"/videos queries=\"{queries}\" k={k}")

    if ">>" in queries:
        queries = [q.strip() for q in queries.split(">>")]
    else:
        queries = [queries.strip()]

    videos, accuracy = search_all_queries(queries, k)
    print(f"/video videos={videos} accuracy={accuracy}")
    return {"videos": videos, "accuracy": accuracy}
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=1111)