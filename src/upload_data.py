import wandb, pandas as pd, os
wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(project="MLops_capstone", job_type="upload-data")
artifact = wandb.Artifact("heart-dataset", type="dataset")
artifact.add_file("D:/Work/Atomcamp/MLops/Project/heart_disease_uci.csv")
run.log_artifact(artifact)
run.finish()