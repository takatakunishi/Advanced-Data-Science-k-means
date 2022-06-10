# 参考
# - k-meansとk-means++を視覚的に理解する~Pythonにてスクラッチから~: Koo, 医療職からデータサイエンティストへ (online), available from <> (accessed 2022-06-10)
# - k-means++を理解する: @g-k, Qiita (online), available from <https://qiita.com/g-k/items/e1d558ffcdc833e6382c> (accessed 2022-06-10)

from tokenize import group
import numpy as np
import fileRead
MAX_LOOP=500
SEED_NUM = 43
N_CULSTER = 6
NAME = [
  "北海道",
  "青森県",
  "岩手県",
  "宮城県",
  "秋田県",
  "山形県",
  "福島県",
  "茨城県",
  "栃木県",
  "群馬県",
  "埼玉県",
  "千葉県",
  "東京都",
  "神奈川県",
  "新潟県",
  "富山県",
  "石川県",
  "福井県",
  "山梨県",
  "長野県",
  "岐阜県",
  "静岡県",
  "愛知県",
  "三重県",
  "滋賀県",
  "京都府",
  "大阪府",
  "兵庫県",
  "奈良県",
  "和歌山県",
  "鳥取県",
  "島根県",
  "岡山県",
  "広島県",
  "山口県",
  "徳島県",
  "香川県",
  "愛媛県",
  "高知県",
  "福岡県",
  "佐賀県",
  "長崎県",
  "熊本県",
  "大分県",
  "宮崎県",
  "鹿児島県",
  "沖縄県"
]

def main():
  print("start")
  data = fileRead.readData()
  minDistance = 10000
  minGroup = [0 for i in range(N_CULSTER)]
  for i in range(100):
    model = KMeans(N_CULSTER)
    distance, group = model.fit(data)
    if distance < minDistance:
      minDistance = distance
      minGroup = group
  print("sum all distance", minDistance)
  print("group result", minGroup)



class KMeans:
  def __init__(self, n_clusters, max_loop = MAX_LOOP, random_seed = SEED_NUM):
    self.n_clusters = n_clusters
    self.max_loop = max_loop
    self.random_state = np.random.RandomState(random_seed)

  def fit(self, X):
    firstIndex = np.array([])
    tmp = np.random.choice(np.array(range(X.shape[0])))
    firstIndex = np.r_[firstIndex, tmp]
    print("tmp",X.shape[1],X,tmp)
    first_cluster = X.loc[tmp]
    print("first_cluster_1", first_cluster)
    # first_cluster = first_cluster[np.newaxis,:]

    # for index, c in X.iterrows():
    #   print("double",c, ((c - first_cluster) **2).sum() /((X - first_cluster)**2).sum())
    print("p", [np.linalg.norm(c - first_cluster) for index, c in X.iterrows()], [np.linalg.norm(c - first_cluster) for index, c in X.iterrows()]/ np.array([np.linalg.norm(c - first_cluster) for index, c in X.iterrows()]).sum())
    p = [np.linalg.norm(c - first_cluster) for index, c in X.iterrows()]/ np.array([np.linalg.norm(c - first_cluster) for index, c in X.iterrows()]).sum()

    r =  np.random.choice(np.array(range(X.shape[0])), size = 1, replace = False, p = p)[0]
    firstIndex = np.r_[firstIndex, r]

    first_cluster = np.r_[[first_cluster],[X.loc[r]]]
    print("cluster",r,first_cluster)
    print("first_cluster.shape",first_cluster.shape[0])
    # distance = [((c - first_cluster) **2).sum() for index, c in X.iterrows() / ((X - first_cluster)**2).sum()]
    # print("distance", distance)
    
    if self.n_clusters >= 3:
      while first_cluster.shape[0] < self.n_clusters:
        dist_f =np.array([sum([np.linalg.norm(x - c) for c in first_cluster]) for index, x in X.iterrows()])
        print("dist_f", dist_f)
        f_argmin = dist_f.argmin()
        print('f_argmin', f_argmin)
        dist_f[f_argmin] = 0
        print("dist_f2", dist_f)

        if not (len(firstIndex) == 0):
          for i in range(len(firstIndex)):
            print("firstIndex",i, firstIndex[i])
            dist_f[int(firstIndex[i])] = 0
        pp = dist_f / dist_f.sum()

        print('pp', pp)
        rr = np.random.choice(np.array(range(X.shape[0])), size=1, replace=False, p = pp)[0]
        if (not rr in firstIndex):
          firstIndex = np.r_[firstIndex, int(rr)]
        else:
          continue
        print("rr", rr, np.array(X.loc[rr]))
        first_cluster = np.r_[first_cluster,[X.loc[rr]]]
        print("new first_cluster", first_cluster)

    dist = np.array([[np.linalg.norm(x - c) for c in first_cluster] for index, x in X.iterrows()])
    print("dist", dist)
    self.labels_ = [d.argmin() for d in dist]
    print('first label', [d.argmin() for d in dist])
    labels_prev = [[0] * X.shape[0]]
    count = 0
    self.cluster_centers_ = [[np.random.random()] * X.shape[1] for i in range(self.n_clusters)]
    print("init", labels_prev, self.cluster_centers_)

    while (not self.labels_ == labels_prev and count < self.max_loop):
      for i in range(self.n_clusters):
        index = [s for s, x in enumerate(self.labels_) if x == i]
        XX = X.iloc[index,:]
        print("index", i)
        print(XX)
        print(np.array(XX.mean()))
        self.cluster_centers_[i] = np.array(XX.mean())
        print("self.cluster_centers_", self.cluster_centers_)
      dist =np.array([[np.linalg.norm(x - c) for c in self.cluster_centers_] for index, x in X.iterrows()])
      print("dist_f", dist)
      f_argmin = dist.argmin()
      # if ((labels_prev == self.labels_).all()):
      #   print("finish")
      #   break
      labels_prev = self.labels_
      self.labels_ = [d.argmin() for d in dist]
      print("tmp result", self.labels_)
      count += 1
    print("result", self.labels_)
    for index in range(len(self.labels_)):
      print(self.labels_[index],NAME[index])
    distanceResult = [0 for i in range(self.n_clusters)]
    groupResult = [0 for i in range(self.n_clusters)]
    for i in range(self.n_clusters):
      index = [s for s, x in enumerate(self.labels_) if x == i]
      regions = [NAME[r] for r in index]
      print("group", i, regions)
      groupResult[i] = regions
      distanceResult[i] = np.array([np.linalg.norm(X.loc[ri] - self.cluster_centers_[i]) for ri in index]).sum()
    print('all distance', distanceResult, firstIndex)
    return sum(distanceResult), groupResult

main()

