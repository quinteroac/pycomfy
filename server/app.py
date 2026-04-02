from fastapi import FastAPI

app = FastAPI(title="parallax-worker", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: add routes (generate, edit, download, ...)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
