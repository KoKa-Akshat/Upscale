import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import os

# Capture & Process Color Dominance
def extract_dominant_color(image_path, n_colors=1):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    # Use KMeans to find dominant color
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img)
    dominant_color = kmeans.cluster_centers_.astype(int)

    return dominant_color[0]

# Complementary Color
def color_complement(color_rgb):
    # Convert RGB to HSV
    color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    complementary_hue = (color_hsv[0] + 180) % 360  # 180 degrees offset for complement

    # Convert back to RGB for display
    complementary_rgb = cv2.cvtColor(
        np.uint8([[[complementary_hue, color_hsv[1], color_hsv[2]]]]), cv2.COLOR_HSV2RGB)[0][0]
    return complementary_rgb

# Example clothing items with image paths
clothing_items = {
    0: {"type": "top1", "color": extract_dominant_color("images/top1.jpg"), "path": "images/top1.jpg"},
    1: {"type": "bottom1", "color": extract_dominant_color("images/bottom1.jpg"), "path": "images/bottom1.jpg"},
    2: {"type": "top2", "color": extract_dominant_color("images/top2.jpg"), "path": "images/top2.jpg"},
    3: {"type": "bottom2", "color": extract_dominant_color("images/bottom2.jpg"), "path": "images/bottom2.jpg"}
}

# Check if colors complement each other
outfit_suggestions = []
for i in range(0, len(clothing_items), 2):
    top_color = clothing_items[i]["color"]
    bottom_color = clothing_items[i + 1]["color"]

    # Check if colors complement each other
    if np.allclose(color_complement(top_color), bottom_color, atol=50):  # Adjust tolerance as needed
        outfit_suggestions.append((clothing_items[i], clothing_items[i + 1]))

# Display Outfit Suggestions using Streamlit
st.title("Suggested Outfits")

for outfit in outfit_suggestions:
    col1, col2 = st.columns(2)
    with col1:
        st.image(outfit[0]["path"], caption="Top")
    with col2:
        st.image(outfit[1]["path"], caption="Bottom")
