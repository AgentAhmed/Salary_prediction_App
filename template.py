import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# project_name="salarypredict"

# list_of_files=[
#     f"src/{project_name}/__init__.py",
#     f"src/{project_name}/components/__init__.py",
#     f"src/{project_name}/components/data_ingestion.py",
#     f"src/{project_name}/components/data_transformation.py",
#     f"src/{project_name}/components/model_tranier.py",
#     f"src/{project_name}/components/model_monitering.py",
#     f"src/{project_name}/pipelines/__init__.py",
#     f"src/{project_name}/pipelines/training_pipeline.py",
#     f"src/{project_name}/pipelines/prediction_pipeline.py",
#     f"src/{project_name}/exception.py",
#     f"src/{project_name}/logger.py",
#     f"src/{project_name}/utils.py",
#     "main.py",
#     "app.py",
#     "Dockerfile",
#     "requirements.txt",
#     "setup.py"
# ]


list_of_files=[
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/components/data_ingestion.py",
    f"src/components/data_transformation.py",
    f"src/components/model_tranier.py",
    f"src/components/model_monitering.py",
    f"src/pipelines/__init__.py",
    f"src/pipelines/training_pipeline.py",
    f"src/pipelines/prediction_pipeline.py",
    f"src/exception.py",
    f"src/logger.py",
    f"src/utils.py",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")