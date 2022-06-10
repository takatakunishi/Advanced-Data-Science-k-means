# 参考
# - k-meansとk-means++を視覚的に理解する~Pythonにてスクラッチから~: Koo, 医療職からデータサイエンティストへ (online), available from <> (accessed 2022-06-10)
# - k-means++を理解する: @g-k, Qiita (online), available from <https://qiita.com/g-k/items/e1d558ffcdc833e6382c> (accessed 2022-06-10)

import numpy as np
import pandas as pd
MAX_LOOP=500
SEED_NUM = 43
N_CULSTER = 2
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
  data = readCSV()
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

def readCSV():
  s = pd.read_csv(
    f"../file/sangyohi.csv",
    encoding="UTF8",
    header=None
  )
  return s

class KMeans:
  def __init__(self, n_clusters, max_loop = MAX_LOOP, random_seed = SEED_NUM):
    self.n_clusters = n_clusters
    self.max_loop = max_loop
    self.random_state = np.random.RandomState(random_seed)

  def fit(self, X):
    firstIndex = np.array([])
    tmp = np.random.choice(np.array(range(X.shape[0])))
    firstIndex = np.r_[firstIndex, tmp]
    first_cluster = X.loc[tmp]
    p = [np.linalg.norm(c - first_cluster) for index, c in X.iterrows()]/ np.array([np.linalg.norm(c - first_cluster) for index, c in X.iterrows()]).sum()
    r =  np.random.choice(np.array(range(X.shape[0])), size = 1, replace = False, p = p)[0]
    firstIndex = np.r_[firstIndex, r]
    first_cluster = np.r_[[first_cluster],[X.loc[r]]]

    if self.n_clusters >= 3:
      while first_cluster.shape[0] < self.n_clusters:
        dist_f =np.array([sum([np.linalg.norm(x - c) for c in first_cluster]) for index, x in X.iterrows()])
        f_argmin = dist_f.argmin()
        dist_f[f_argmin] = 0

        if not (len(firstIndex) == 0):
          for i in range(len(firstIndex)):
            dist_f[int(firstIndex[i])] = 0
        pp = dist_f / dist_f.sum()

        rr = np.random.choice(np.array(range(X.shape[0])), size=1, replace=False, p = pp)[0]
        if (not rr in firstIndex):
          firstIndex = np.r_[firstIndex, int(rr)]
        else:
          continue
        first_cluster = np.r_[first_cluster,[X.loc[rr]]]

    dist = np.array([[np.linalg.norm(x - c) for c in first_cluster] for index, x in X.iterrows()])
    self.labels_ = [d.argmin() for d in dist]
    labels_prev = [[0] * X.shape[0]]
    count = 0
    self.cluster_centers_ = [[np.random.random()] * X.shape[1] for i in range(self.n_clusters)]

    while (not self.labels_ == labels_prev and count < self.max_loop):
      for i in range(self.n_clusters):
        index = [s for s, x in enumerate(self.labels_) if x == i]
        XX = X.iloc[index,:]
        self.cluster_centers_[i] = np.array(XX.mean())
      dist =np.array([[np.linalg.norm(x - c) for c in self.cluster_centers_] for index, x in X.iterrows()])
      f_argmin = dist.argmin()
      labels_prev = self.labels_
      self.labels_ = [d.argmin() for d in dist]
      count += 1
    distanceResult = [0 for i in range(self.n_clusters)]
    groupResult = [0 for i in range(self.n_clusters)]
    for i in range(self.n_clusters):
      index = [s for s, x in enumerate(self.labels_) if x == i]
      regions = [NAME[r] for r in index]
      groupResult[i] = regions
      distanceResult[i] = np.array([np.linalg.norm(X.loc[ri] - self.cluster_centers_[i]) for ri in index]).sum()
    return sum(distanceResult), groupResult

main()

