from fastapi import FastAPI, Request

# from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.responses import JSONResponse, RedirectResponse

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://thc-test-santhosh.azurewebsites.net/",
    ],  # Replace with the actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/api/hello")
async def get_inference(request: Request):
    # import pdb

    # pdb.set_trace()
    return {"message": "Hello world"}


@app.get("/{full_path:path}")
async def catch_all(request: Request):

    ctx = {
        "request": request,
    }
    return {"catchingAll": "Hello Catch"}
