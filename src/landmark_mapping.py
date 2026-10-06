import numpy as np

def _distance(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def map_landmarks_to_blendshapes(landmarks):
    lm13=landmarks["lm13"]
    lm14=landmarks["lm14"]
    mouth_open=_distance(lm13,lm14)
    mouth_open=min(max(mouth_open*2.5,0),1)
    return {"mouth_open":mouth_open}