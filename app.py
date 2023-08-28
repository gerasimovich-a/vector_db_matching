from fastapi import FastAPI
import uvicorn
from typing import Union

import faiss
import numpy as np
import joblib

from scipy.spatial import distance as dist

app = FastAPI()

faiss_index = None
model = None
scaler = None
base_array = None
mask = np.array([
    True, True, True, True, True, True, False, True, True,
    True, True, True, True, True, True,  True, True, True,
    True, True, True, False, True, True,  True, False, True,
    True, True, True, True, True, True, False, True, True,
    True, True, True, True, True, True, True, True, False,
    True, True, True, True, True, True, True, True, True,
    True, True, True, True, True, False, True,  True, True,
    True, True, False, True, True, True,  True, False, True])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, 'datasets/')


def parse_string(vec: str) -> list[float]:
    """
    1.23,6.7 -> [1.23 6.7]
    :param vec:
    :return:
    """
    val_list = vec.split(",")
    if len(val_list) != dims:
        return None
    return [float(el) for el in val_list]

def get_distances(vec_1, vec_2):
    '''
    принимает на вход 2 вектора, возвращает список расстояний
    '''
    distances = [
        dist.braycurtis(vec_1,vec_2),
        dist.canberra(vec_1,vec_2),
        dist.chebyshev(vec_1,vec_2),
        dist.cityblock(vec_1,vec_2), #Manhattan
        dist.correlation(vec_1,vec_2),
        dist.cosine(vec_1,vec_2),
        dist.euclidean(vec_1,vec_2),
        dist.minkowski(vec_1,vec_2,3),
        dist.minkowski(vec_1,vec_2,5)
    ]
    return distances


def make_concatenated_with_distances(candidates, query, base):

    concatenated_vecs = []
    vec_1 = query
    for candidate in candidates:
        vec_2 = base[candidate]
        concatenated_vec = np.concatenate([vec_1,vec_2])
        distances = get_distances(vec_1[mask], vec_2[mask])
        concatenated_vec = np.concatenate([concatenated_vec,distances])
        concatenated_vecs.append(concatenated_vec)
    return np.array(concatenated_vecs)


@app.on_event("startup")
def start():
    global faiss_index
    global model
    global scaler
    global base_array
    
    
    #загружаем подготовленный индекс
    faiss_index = read_index(os.path.join(PATH, 'faiss_index.index'))
    #можно изменить на большее число и получить метрику выше, но ниже скорость работы
    index.nprobe = 32
    
    model = cb.CatBoostClassifier()
    model.load_model(os.path.join(PATH, 'cb_model'))
    
    scaler = joblib.load(os.path.join(PATH, 'std_scaler.bin'))
    
    base_array = np.load(os.path.join(PATH, 'datasets/base_array.npy'))


@app.get("/")
def main() -> dict:
    return {"status": "OK", "message": "Введите вектор"}


@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    global faiss_index
    global model
    global scaler
    global base_array
    
    if item is None:
        return {"status": "fail", "message": "No input data"}

    vec = parse_string(item)
    vec = scaler.transform(vec)
    vec = np.ascontiguousarray(vec, dtype="float")[np.newaxis, :]
    _, idxs = faiss_index.search(vec, k=100)
    
    query = vec
    query_features = make_concatenated_with_distances([idxs], query, base_array)

    return {"status": "OK", "data": [str(el) for el in idx]}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)