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
  # cluster_centroid = [[np.random.random()] * data.shape[1] for i in range(n_cluster)]
  cluster_centroid = initCluster(originalData, data, clusters, n_cluster)

  print("cluster_centroid", cluster_centroid)
  for tryNumber in range(max_loop):
    dataWithCluster = originalData.copy()
    dataWithCluster[CLUSTERNAME] = clusters
    # print("dataWithCluster", dataWithCluster)
    for n in range(n_cluster):
      cluster_centroid[n] = dataWithCluster[dataWithCluster[CLUSTERNAME] == n].drop([CLUSTERNAME], axis=1).mean(axis = 0).array
      # print(dataWithCluster[dataWithCluster[CLUSTERNAME] == n].drop([CLUSTERNAME], axis=1).mean())
      # print('range n = ',n)
    # for c in cluster_centroid:
    #   print("c try")
    #   print(np.linalg.norm(originalData - c, axis= 1))
    # print(np.array([np.linalg.norm(originalData - c, axis= 1) for c in cluster_centroid]).argmin(axis=0))

    # 代表ベクトルまでのユークリッド距離で分類
    new_cluster = np.array([np.linalg.norm(originalData - c, axis= 1) for c in cluster_centroid]).argmin(axis=0)
    print("new_cluster", new_cluster)
    for n in range(n_cluster):
      if not np.any(new_cluster == n):
        cluster_centroid[n] = originalData[np.random.choice(data.shape[0], 1), :]
    # print("cluster_centroid", cluster_centroid)
    # 収束判定
    if np.allclose(clusters, new_cluster):
      break
    clusters = new_cluster
    print(tryNumber)

  print("result", clusters[0], clusters[25], clusters[46])
  print("mikke")

def initCluster(originalData, data, clusters, n_cluster):
  doInit = True
  cluster_centroid = [[np.random.random() for j in range(data.shape[1]) ] for i in range(n_cluster)]
  print("cluster_centroid", cluster_centroid)
  print("originalData", originalData)
  while doInit:
    doInit = False
    dataWithCluster = (originalData.copy()).values
    # dataWithCluster[CLUSTERNAME] = clusters
    # print("dataWithCluster", dataWithCluster)
    # for n in range(n_cluster):
    #   cluster_centroid[n] = dataWithCluster[dataWithCluster[CLUSTERNAME] == n].drop([CLUSTERNAME], axis=1).mean(axis = 0).array
      # print(dataWithCluster[dataWithCluster[CLUSTERNAME] == n].drop([CLUSTERNAME], axis=1).mean())
      # print('range n = ',n)
    # for c in cluster_centroid:
    #   print("c try")
    #   print(np.linalg.norm(originalData - c, axis= 1))
    # print(np.array([np.linalg.norm(originalData - c, axis= 1) for c in cluster_centroid]).argmin(axis=0))

    # 代表ベクトルまでのユークリッド距離で分類
    for c in cluster_centroid:
      print("try", c, [np.linalg.norm(dataWithCluster - c, axis= 1)])
    new_cluster = np.array([np.linalg.norm(dataWithCluster - c, axis= 1) for c in cluster_centroid]).argmin(axis=0)
    print("new_cluster tes", np.array([np.linalg.norm(dataWithCluster - c, axis= 1) for c in cluster_centroid]), new_cluster)
    for n in range(n_cluster):
      if not np.any(new_cluster == n):
        # doInit = True
        cluster_centroid = [[np.random.random() for j in range(data.shape[1]) ] for i in range(n_cluster)]
        # print("cluster_centroid", cluster_centroid)
  return cluster_centroid

def test():
  # seed値固定
  np.random.seed(874)
  # x座標
  x = np.r_[np.random.normal(size=1000,loc=0,scale=1),
            np.random.normal(size=1000,loc=4,scale=1)]
  # y座標
  y = np.r_[np.random.normal(size=1000,loc=10,scale=1),
            np.random.normal(size=1000,loc=10,scale=1)]
  data = np.c_[x, y]
  # クラスタ数
  n_clusters = 2
  # 最大ループ数
  max_iter = 300
  # 所属クラスタ
  clusters = np.random.randint(0, n_clusters, data.shape[0])
  for _ in range(max_iter):
    # 中心点の更新
    # 各クラスタのデータ点の平均をとる
    centroids = np.array([data[clusters == n, :].mean(axis = 0) for n in range(n_clusters)])

    # 所属クラスタの更新
    # 一番近い中心点のクラスタを所属クラスタに更新する
    # np.linalg.normでノルムが計算できる
    # argminで最小値のインデックスを取得できる
    new_clusters = np.array([np.linalg.norm(data - c, axis = 1) for c in centroids]).argmin(axis = 0)

    # 空のクラスタがあった場合は中心点をランダムな点に割り当てなおす
    for n in range(n_clusters):
        if not np.any(new_clusters == n):
            centroids[n] = data[np.random.choice(data.shape[0], 1), :]

    # 収束判定
    if np.allclose(clusters, new_clusters):
        break

    clusters = new_clusters
main()

