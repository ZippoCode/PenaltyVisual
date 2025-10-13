import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("penaltyvision").project("penalty-kick-visual-tfbmd")
version = project.version(4)

save_path = "/home/sprochilo/ai_project/PenaltyVisual/data/penalty_kick"
os.makedirs(save_path, exist_ok=True)
dataset = version.download("yolov11", location=save_path, overwrite=True)
