import numpy as np
import util
import fileRead
# https://qiita.com/navitime_tech/items/bb1bd01537bc2713444a#%E6%9C%80%E5%88%9D%E3%81%AB
# https://qiita.com/g-k/items/e1d558ffcdc833e6382c
MAX_LOOP=500
SEED_NUM = 43

def main():
  print("start")
  data = fileRead.readData()
  model = KMeans(3)
  model.fit(data)

class KMeans:
  def __init__(self, n_clusters, max_loop = MAX_LOOP, random_seed = SEED_NUM):
    self.n_clusters = n_clusters
    self.max_loop = max_loop
    self.random_state = np.random.RandomState(random_seed)

  def sq(data):
    print("double",data, ((data - first_cluster) **2).sum())

  def fit(self, X):
    tmp = np.random.choice(np.array(range(X.shape[0])))
    print("tmp",X,tmp)
    first_cluster = X.loc[tmp]
    print("first_cluster_1", first_cluster)
    # first_cluster = first_cluster[np.newaxis,:]

    for index, c in X.iterrows():
      print("double",c, ((c - first_cluster) **2).sum())

    distance = [((c - first_cluster) **2).sum() for index, c in X.iterrows()]
    print("distance", distance)

main()

