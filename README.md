# Seismic-CAE-RF
Machine learning workflow of convolutional autoencoder and random forest to identify seismic facies and predict sandstone thickness distribution.


1.make-dataset.py make seismic data converted to a gray-scale image and the gray-scale image is split into patches by the given patch cell.

2.CAEL1.py make patches split into training and test sets and provided as input and output to CAE for training.

3.CAEPre.py  make the seismic section sent to the encoder of the CAE to obtain a latent eigenvalue matrix.

4.PCA-Kmeans.py: PCA is applied to extract the principal components (PCs) to form a set of low-dimensional latent eigenvalues. 

The PCs as the low-dimensional latent eigenvalues, are provided to K-means clustering to generate the seismic facies classification.
