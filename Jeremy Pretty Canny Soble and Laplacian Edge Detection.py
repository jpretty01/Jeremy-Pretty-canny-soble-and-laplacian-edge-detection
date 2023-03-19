# Jeremy Pretty
# Discussion Board 7
import cv2
import numpy as np

def generate_image():
    image = np.zeros((256, 256), dtype=np.uint8)

    # Draw a filled-in square
    cv2.rectangle(image, (50, 50), (100, 100), 200, -1)

    # Draw a filled-in circle
    cv2.circle(image, (200, 200), 25, 150, -1)

    return image

def add_noise(image, mean=0, std_dev=30):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_edge_detection(image):
    # Canny
    canny = cv2.Canny(image, 100, 200)

    # Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y).astype(np.uint8)

    # Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.abs(laplacian).astype(np.uint8)

    return canny, sobel, laplacian

def evaluate_edge_detection(image, edge_image):
    true_edges = np.zeros_like(image, dtype=bool)
    true_edges[48:102, 48:102] = 1
    true_edges[175:225, 175:225] = 1

    detected_edges = edge_image > 0

    tp = np.sum(np.logical_and(true_edges, detected_edges))
    fp = np.sum(np.logical_and(np.logical_not(true_edges), detected_edges))
    fn = np.sum(np.logical_and(true_edges, np.logical_not(detected_edges)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

if __name__ == "__main__":
    image = generate_image()
    noisy_image = add_noise(image)

    for img, name in [(image, "Original Image"), (noisy_image, "Noisy Image")]:
        canny, sobel, laplacian = apply_edge_detection(img)

        print(f"Evaluation for {name}:")
        for edge_image, method in [(canny, "Canny"), (sobel, "Sobel"), (laplacian, "Laplacian")]:
            precision, recall, f1_score = evaluate_edge_detection(img, edge_image)
            print(f"{method}: Precision={precision:.2f}, Recall={recall:.2f}, F1 Score={f1_score:.2f}")

        print()
