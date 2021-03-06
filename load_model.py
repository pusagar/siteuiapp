import joblib
import requests
import os

def get_model(model_path):
    print(model_path)
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")   
        if not 'randomforestmodel.pkl' in os.listdir('.'):
            # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download&confirm=t"
            #url = "https://drive.google.com/u/0/uc?id=1eZQFdqwAMHLsjwD2C-PyvcZlVAwmAudD&export=download&confirm=t"
            url = "https://drive.google.com/u/0/uc?id=1jodlo5KYy7f-ug8O4r4uMnuhLpVwyHax&export=download&confirm=t"
            r = requests.get(url, allow_redirects=True)
            open(r"randomforestmodel.pkl", 'wb').write(r.content)
            del r
        with open(r"randomforestmodel.pkl", "rb") as m:
            rf = joblib.load(m)
    return rf

