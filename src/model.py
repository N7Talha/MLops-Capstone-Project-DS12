from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

numeric = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

pre = ColumnTransformer([
        ('num', Pipeline(steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
        ]), numeric),
        ('cat', Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical)
])
clf = LogisticRegression(max_iter=1000, random_state=42)
model = Pipeline(steps=[('prep', pre), ('clf', clf)])

def build_model():
    return model

def save_model(pipe, path):
    joblib.dump(pipe, path)

def load_model(path):
    return joblib.load(path)