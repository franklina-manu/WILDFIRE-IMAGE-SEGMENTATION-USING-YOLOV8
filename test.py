import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="0gvK5Tu6B6V2daODNXoP"
)

result = CLIENT.infer("fire.png", model_id="fire-smoke-yolov8/1")

print(result)


image = cv2.imread("fire.png")

# Loop through each prediction and draw the bounding box
for prediction in result["predictions"]:
    x_center = prediction["x"]
    y_center = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]
    confidence = prediction["confidence"]
    label = prediction["class"]

    # Convert center x, y, width, and height to top-left and bottom-right corners
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Draw rectangle (bounding box)
    color = (
        (0, 0, 255) if label == "Fire" else (255, 0, 0)
    )  # Red for fire, blue for smoke
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Add label and confidence
    label_text = f"{label} {confidence:.2f}"
    cv2.putText(
        image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
    )

# Display the image with bounding boxes
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the image with bounding boxes
cv2.imwrite("fire_detected.png", image)
