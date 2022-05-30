import numpy as np
import util
import fileRead

CLUSTERNAME = "culster"

def main():
  data = fileRead.readData()
  kMeans(data)
  util.hoge()

def kMeans(data):
  n_cluster = 6
  max_loop = 500
  clusters = np.random.randint(0, n_cluster, data.shape[0])
  originalData = data
  print(CLUSTERNAME, data.shape[0], clusters)
  cluster_centroid = [[0] * originalData.shape[1] for i in range(n_cluster)]

  # print("cluster_centroid", cluster_centroid)
  for _ in range(max_loop):
    dataWithCluster = originalData.copy()
    dataWithCluster[CLUSTERNAME] = clusters
    # print("dataWithCluster", dataWithCluster)
    for n in range(n_cluster):
      cluster_centroid[n] = dataWithCluster[dataWithCluster[CLUSTERNAME] == n].drop([CLUSTERNAME], axis=1).mean(axis = 0).array
      print(dataWithCluster[dataWithCluster[CLUSTERNAME] == n].drop([CLUSTERNAME], axis=1).mean())
      print('range n = ',n)
    # for c in cluster_centroid:
    #   print("c try")
    #   print(np.linalg.norm(originalData - c, axis= 1))
    print(np.array([np.linalg.norm(originalData - c, axis= 1) for c in cluster_centroid]).argmin(axis=0))
    new_cluster = np.array([np.linalg.norm(originalData - c, axis= 1) for c in cluster_centroid]).argmin(axis=0)
    for n in range(n_cluster):
      if not np.any(new_cluster == n):
        cluster_centroid[n] = data[np.random.choice(data.shape[0], 1), :]
    print("cluster_centroid", cluster_centroid)
    # 収束判定
    if np.allclose(clusters, new_cluster):
      break
    clusters = new_cluster

  print("result", clusters[0], clusters[25], clusters[46])
  print("mikke")

main()

