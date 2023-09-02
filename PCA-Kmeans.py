import mglearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



classnumber = 10 #category
z_true = np.load(r'z3-z4pre.npy')  #Load the extracted latent eigenvalues
pca = PCA(n_components=0.3,whiten=True,svd_solver='auto') #Use default parameters
components = z_true
kmeans = KMeans(n_clusters=classnumber, random_state=0).fit(components)
labels = kmeans.labels_
labels = labels.reshape(485, 271).T
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=80)
bounds = list(range(classnumber+1))
colors = ['#800000','#B22222','#FF0000','#FF4500','#8B4513','#FF8C00','#FFFF00','#ADFF2F','#90EE90','#7FFFD4','#00BFFF','#00008B','#0000FF','#FFFFFF','#FF0000','#F08080','#6495ED','#FFFF00','#FFA500','#808080','#FF8C00','#6B8E23','#800000','#B22222']
cmap = ListedColormap(colors)
norms = BoundaryNorm(bounds, cmap.N)
plt.imshow(labels,cmap=cmap,norm=norms,aspect='auto')
plt.colorbar(shrink=1)
plt.show()