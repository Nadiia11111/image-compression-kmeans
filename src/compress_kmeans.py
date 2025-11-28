import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = plt.imread('data/plant.png')

# Visualize the image
plt.imshow(image)
plt.title("Our image")
plt.axis('off')
plt.show()
plt.close()

print("Shape of image is:", image.shape)

# Reshape the 3D image array (height, width, 3) into a 2D array (num_pixels, 3)
# Each row now represents a single pixel in RGB format, which is required for K-Means clustering.
X_img = np.reshape(image, (image.shape[0] * image.shape[1], 3))


# Initialize K random centroids by shuffling the data and selecting the first K points.
def kMeans_random_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    
    return centroids


K = 16    # quantity of centroids
max_iters = 10

initial_centroids = kMeans_random_centroids(X_img, K)


# Find closest centroids
# Compute the squared Euclidean distance between every point and every centroid
def find_closest_centroids(X, centroids):
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    
    # For each point choose the index of the closest centroid
    return np.argmin(distances, axis=1)


# Compute new centroids by taking the mean of all data points assigned to each cluster
def compute_centroids(X, indexes, K):
    m, n = X.shape
    centroids = np.zeros((K,n))
    
    for i in range(K):
        point = X[indexes == i]
        centroids[i] = np.mean(point, axis=0)
        
    return centroids


# Run the K-Means algorithm by repeatedly assigning points to the nearest
# centroid and updating centroid positions for a fixed number of iterations
def kMeans(X, initial_centroids, max_iters=10):
    m, n = X.shape
    K = initial_centroids.shape[0]    # quantity of centroids
    centrds = initial_centroids
    prev_centrds = centrds
    indexes = np.zeros(m)
    
    for i in range(max_iters):
        print(f'Iteration {i+1}/{max_iters}')
        indexes = find_closest_centroids(X, centrds)
        centrds = compute_centroids(X, indexes, K)
        
    return centrds, indexes
    

# Run K-Means
centroids, indexes = kMeans(X_img, initial_centroids, max_iters)

# Replace each pixel with its centroid color
compressed_img = centroids[indexes].reshape(image.shape)

# Visualize the compressed image
plt.imshow(compressed_img)
plt.axis('off')
plt.title("Image after compression")
plt.show()
plt.close()

plt.imsave('data/compressed_k16.png', compressed_img)


# Viualize difference between origin image and compressed one
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis('off')

# Compressed image
plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title(f"Compressed (K={K})")
plt.axis('off')

plt.tight_layout()
plt.show()