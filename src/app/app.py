from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
from src.av_randlanet_scfnet import main_OrchardRoad


app = FastAPI(title="API for Geospatial AI")


class PointCloud(BaseModel):
    pcd_url: str


@app.get("/")
async def root():
    return {"message": "Hello GeoAI!"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


@app.post("/upload_chunk")
def upload_chunk(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


@app.post("/upload_multiple")
def upload_multiple(files: List[UploadFile] = File(...)):
    for file in files:
        try:
            contents = file.file.read()
            with open(file.filename, 'wb') as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            file.file.close()

    return {"message": f"Successfuly uploaded {[file.filename for file in files]}"}


@app.post("/predict", status_code=200)
async def predict(request: PointCloud):
    prediction = main_OrchardRoad.main(request.pcd_filename)
    if not prediction:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(
            status_code=404, detail="point cloud could not be downloaded"
        )
    return prediction
