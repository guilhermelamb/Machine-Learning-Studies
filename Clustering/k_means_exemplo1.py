import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base = np.array([[20,1000], [27,1200],[21,2900], [37,1850], [46,900], [53,950],
                 [55,2000], [47,2100], [52,3000], [32,5900], [39,4100], [41,5100],
                 [39,7000], [48,5000], [48,6500]])
scaler = StandardScaler()
base = scaler.fit_transform(base)

kmeans = KMeans(n_clusters = 3)

kmeans.fit(base)

centroide = kmeans.cluster_centers_

rotulos = kmeans.labels_

cores = ['g.', 'r.', 'b.']


for i in range(len(base)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize = 15)
plt.scatter(centroide[:, 0], centroide[:,1], marker='x')