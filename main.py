# pylint: disable=missing-class-docstring
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from dataset.configuration_manager import ConfigManager
from logger.manager import LoggerManager

from training.train_unet import UnetTrainWrapper, UnetInfererWrapper
from training.train_custom import CustomTrainWrapper, CustomInfererWrapper

app = FastAPI()

# API Endpoint to Get Configuration by Name
@app.get("/config/{config_name}")
async def read_config(config_name: str):
    logger.log(f"config_name: {config_name}")
    if config_name == "dataset":
        return ConfigManager.fetch_dataset_config()
    elif config_name == "dataloader":
        return ConfigManager.fetch_dataloader_config()
    else:
        raise HTTPException(status_code=404, detail="Config not found")

@app.post("/model/baseline/train")
async def train():
    await UnetTrainWrapper.run()
    return {"message": "Training started"}

@app.post("/model/baseline/infer")
async def infer(image: UploadFile = File(...)):
    
    with open("temp_image.png", "wb") as f:
        f.write(image.file.read())
    
    prediction = UnetInfererWrapper.run(image_path="temp_image.png")
    
    return {"prediction": prediction.tolist()}

@app.post("/model/custom/train")
async def train():
    await CustomTrainWrapper.run()
    return {"message": "Training started"}

@app.post("/model/custom/infer")
async def infer(image: UploadFile = File(...)):
    
    with open("temp_image.png", "wb") as f:
        f.write(image.file.read())
    
    prediction = CustomTrainWrapper.run(image_path="temp_image.png")
    
    return {"prediction": prediction.tolist()}

# Run the FastAPI application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger = LoggerManager().get_logger()
