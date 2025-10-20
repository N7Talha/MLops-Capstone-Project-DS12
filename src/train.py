import wandb, joblib, pandas as pd, os, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.model import build_model, save_model


# 1. config
with open("configs/train_config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. fetch dataset artifact
run = wandb.init(project="MLops_capstone", job_type="train", config=cfg)
artifact = run.use_artifact('heart-dataset:latest', type='dataset')
df_path = artifact.download()
df = pd.read_csv(f"{df_path}/heart_disease_uci.csv")

# 3. split
label_col = 'num'                 
X = df.drop(label_col, axis=1)
y = df[label_col]
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. train
model = build_model()
model.fit(Xtr, ytr)

# 5. evaluate
pred = model.predict(Xval)
acc = accuracy_score(yval, pred)
f1  = f1_score(yval, pred, average='weighted')
print("accuracy", acc)
wandb.log({"accuracy": acc, "f1": f1})

# 6. save model locally
os.makedirs("outputs", exist_ok=True)
save_model(model, "outputs/model.joblib")

# 7. log model as artifact
model_art = wandb.Artifact("sk-model", type="model", metadata={"acc":acc})
model_art.add_file("outputs/model.joblib")
run.log_artifact(model_art)
run.finish()