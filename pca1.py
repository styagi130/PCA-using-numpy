import numpy as np
np.random.seed(1)
mu_vec1 = np.array([0,0,0])  # sample mean
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # sample covariance
#Transpose of a Matrix. A matrix which is 
#formed by turning all the rows of a given matrix into columns and vice-versa. 
#convenience, for printing
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
mu_vec2 = np.array([1,1,1]) # sample mean
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #sample covariance
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
from matplotlib import pyplot as plt
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
#mean for each feature
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

#3D mean vector
mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
#print('Mean Vector:\n', mean_vector)
cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
#print('Covariance Matrix:\n', cov_mat)
eig_val_sc, eig_vec_sc = np.linalg.eig(cov_mat)
for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
#    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
 #   print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
             for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
#print eig_pairs
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#for i in eig_pairs:
#	print(i)

print eig_pairs[1],"\n",eig_pairs[0][1]

matrix_w = np.hstack((eig_pairs[1][1],
                      eig_pairs[2][1]))
#print('Matrix W:\n', matrix_w)

matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
                      eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)

transformed = matrix_w.T.dot(all_samples)
#print (transformed)
assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."


plt.plot(transformed[0,0:20], transformed[1,0:20],
         'o', markersize=7, color='green', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40],
         '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()
