import json
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from service import search_all_queries

app = FastAPI()

app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

with open("config.json", "r") as f:
    config = json.load(f)
    

    
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
    uvicorn.run(app, host="127.0.0.1", port=1111)
