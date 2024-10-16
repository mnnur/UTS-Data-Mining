import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle

def resize_image(image, new_width=None, new_height=None):
    """
    Resizes an image to the specified dimensions while maintaining aspect ratio.

    Args:
        image: The input image as a NumPy array.
        new_width: The desired width of the resized image. If None, the width is calculated to maintain aspect ratio.
        new_height: The desired height of the resized image. If None, the height is calculated to maintain aspect ratio.

    Returns:
        The resized image as a NumPy array.
    """
    height, width = image.shape[:2]

    if new_width is None and new_height is None:
        return image

    if new_width is None:
        new_width = int(new_height * width / height)
    elif new_height is None:
        new_height = int(new_width * height / width)

    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img

def calculate_distances(pixels, centroids):
  """
  Calculates distances between pixels and cluster centroids.

  Args:
      pixels: A NumPy array representing the image pixels.
      centroids: The cluster centroids.

  Returns:
      A NumPy array of distances between each pixel and all centroids.
  """
  distances = np.linalg.norm(pixels[:, None] - centroids, axis=2)
  return distances

def inference(image, centroids):

    # Reshape the new image pixels
    new_image_pixels = image.reshape(-1, 3)

    # Calculate distances to centroids and assign labels
    distances = calculate_distances(new_image_pixels, centroids)
    new_image_labels = np.argmin(distances, axis=1)

    # Reshape labels to image dimensions
    new_image_labels = new_image_labels.reshape(image.shape[:2])

    # Assign colors to clusters
    colors = assign_colors(centroids)

    # Create segmented image
    segmented_new_image = create_segmented_image(image.copy(), new_image_labels, colors)
    
    return segmented_new_image

def kmeans_clustering(image, k):
    """
    Performs K-means clustering on the pixel colors of an image.

    Args:
        image: The input image as a NumPy array.
        k: The number of clusters.

    Returns:
        A tuple containing:
            - The cluster labels for each pixel.
            - The cluster centroids.
    """
    
    if k == 3:
        with open('modelk3.pkl', 'rb') as f:
            centroids = pickle.load(f)
            
        return inference(image, centroids)
    elif k == 4:
        with open('modelk4.pkl', 'rb') as f:
            centroids = pickle.load(f)
            
        return inference(image, centroids)
    elif k == 5:
        with open('modelk5.pkl', 'rb') as f:
            centroids = pickle.load(f)
            
        return inference(image, centroids)
    elif k == 6:
        with open('modelk6.pkl', 'rb') as f:
            centroids = pickle.load(f)
            
        return inference(image, centroids)

def assign_colors(centroids):
    """
    Assigns colors to the clusters based on a colormap.

    Args:
        centroids: The cluster centroids as a NumPy array.

    Returns:
        A list of colors corresponding to each cluster.
    """
    num_clusters = centroids.shape[0]
    color_map = plt.cm.viridis(np.linspace(0, 1, num_clusters))  # Choose a colormap
    return color_map

def create_segmented_image(image, labels, colors):
    """
    Creates a segmented image by assigning colors to each cluster and adding labels.

    Args:
        image: The original image as a NumPy array.
        labels: The cluster labels for each pixel.
        colors: A list of colors for each cluster.

    Returns:
        The segmented image with assigned colors and labels.
    """
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    # Loop through each unique label
    for label in np.unique(labels):
        # Assuming you have a cluster color defined as a tuple
        cluster_color = (colors[label][:3] * 255).astype(np.uint8)
        
        # Fill pixels with corresponding color
        segmented_image[labels == label] = cluster_color

    for label in np.unique(labels):
        # Get coordinates of pixels in this cluster
        cluster_pixels = np.where(labels == label)

        # Find the majority area by counting pixels in each row and column
        row_counts = np.bincount(cluster_pixels[0])
        col_counts = np.bincount(cluster_pixels[1])

        # Find the indices of the majority row and column
        majority_row_index = np.argmax(row_counts)
        majority_col_index = np.argmax(col_counts)

        # Calculate average location for placing the label in the majority area
        y_center = majority_row_index
        x_center = majority_col_index

        # Pass the cluster color as a tuple to putText
        cv2.putText(segmented_image, f"{label+1}", (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return segmented_image

def visualize(image, k):
    """
    Visualizes the original and segmented images using Streamlit.

    Args:
        image: The input image as a NumPy array.
    """
    # Resize the image to the desired dimensions
    image = resize_image(image, 512, 512)

    # Perform K-means clustering
    segmented_image = kmeans_clustering(image, k)

    # Convert images from BGR to RGB format for display
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    # Display images using Streamlit
    st.image(original_image_rgb, caption="Original Image", use_column_width=True)
    st.image(segmented_image_rgb, caption="Segmented Image", use_column_width=True)


st.title("Image Segmentation using K-Means Clustering")

df = pd.DataFrame({
    'Nama': ["Fakhri Fajar Ramadhan", "Hudzaifah Al Mutaz Billah", "Muhammad Naufal Nur Ramadhan"],
    'NPM': ["140810210046", "140810210050", "140810210058"]
})

st.table(df)

st.markdown("## Choose number of clusters (K) : ")

n_cluster = st.slider('cluster', 3, 6)

image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
    # Read the file from Streamlit's file uploader
    image_bytes = image_file.read()

    # Convert bytes data to a NumPy array for OpenCV processing
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Visualize the original and segmented images
    visualize(image, n_cluster)
