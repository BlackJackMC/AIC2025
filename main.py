import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from service import search_all_queries

app = FastAPI()

app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

@app.get('/segments/{segment}')
async def get_segment(segment: str):
    return FileResponse(f"./dataset/segments/{segment}.mp4", 
                        media_type="video/mp4", 
                        headers= {
                            "Accept-Ranges": "bytes"
                        })

    
@app.get('/videos')
def search(queries: str, k: int = 1):
    print(f"/videos queries=\"{queries}\" k={k}")
    if ">>" in queries:
        queries = [q.strip() for q in queries.split(">>")]
    
    videos, accuracy = search_all_queries(queries, k)
    
    print(f"/video videos={videos} accuracy={accuracy}")
    
    return {
        "videos": videos,
        "accuracy": accuracy
    }
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=1111)
