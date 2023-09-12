import pickle
import xgboost as xgb
from fastapi import FastAPI, Request
import json


with open('models/DT.bin', 'rb') as f_in:
    dv, clf_dt = pickle.load(f_in)

with open('models/RF.bin', 'rb') as f_in:
    dv, clf_rf = pickle.load(f_in)

with open('models/XGB.bin', 'rb') as f_in:
    dv, clf_xgb = pickle.load(f_in)

app = FastAPI()

@app.post('/dt')
async def predict_dt(request: Request):
    customer = await request.json()
    customer = json.loads(customer)
    x = dv.transform([customer])
    pred = clf_dt.predict(x)[0]
    res = {
        "res": int(pred)
    }

    return json.dumps(res)

@app.post('/rf')
async def predict_rf(request: Request):
    customer = await request.json()
    customer = json.loads(customer)
    x = dv.transform([customer])
    pred = clf_rf.predict(x)[0]
    res = {
        "res": int(pred)
    }

    return json.dumps(res)

@app.post('/xgb')
async def predict_xgb(request: Request):
    customer = await request.json()
    customer = json.loads(customer)
    x = dv.transform([customer])
    features = dv.get_feature_names_out()
    x = xgb.DMatrix(x, feature_names=features)
    pred = clf_xgb.predict(x)[0]
    res = {
        "res": int(pred)
    }

    return json.dumps(res)

