from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
import numpy as np

vectors = np.array(
    [[-1, 1, 1],
     [-1, 1, 1],
     [-1, 1, 1],
     [1, 2, 0],
     [1, 0, 2],
     [5, -2, 5],
     [6, 2, 4],
     [3, 0, 3],
     [3, -3, -2],
     [4, 2, -2],
     [0, 2, 0],
     [0, -5, 0],
     [0, -2, 0]], dtype=np.float)
labels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])

cpn = GroupRepelFeatureTransformer(3, 2)

cpn.fit([vectors], [labels])
