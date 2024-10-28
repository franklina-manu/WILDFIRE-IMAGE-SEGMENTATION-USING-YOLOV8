import tensorflow as tf
from keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import glob
import matplotlib.pyplot as plt

# Directories for train, validation, and test data
train_data_dir = "./data/FlameVision/Classification/train"
val_data_dir = "./data/FlameVision/Classification/valid"
test_data_dir = "./data/FlameVision/Classification/test"

# Image dimensions and number of classes
img_width, img_height = 416, 416
num_classes = 2

# Count images for Fire and No Fire categories
train_nofire = glob.glob(os.path.join(train_data_dir, "nofire/*png"))
train_fire = glob.glob(os.path.join(train_data_dir, "fire/*png"))
val_nofire = glob.glob(os.path.join(val_data_dir, "nofire/*png"))
val_fire = glob.glob(os.path.join(val_data_dir, "fire/*png"))
test_nofire = glob.glob(os.path.join(test_data_dir, "nofire/*png"))
test_fire = glob.glob(os.path.join(test_data_dir, "fire/*png"))

# Visualize image distribution
categories = ["Fire", "No Fire"]
num_images = [
    len(train_fire) + len(val_fire) + len(test_fire),
    len(train_nofire) + len(val_nofire) + len(test_nofire),
]
custom_colors = ["#ffb366", "#c2f0f0"]

# Check if there are any images before attempting to create a pie chart
if sum(num_images) > 0:
    plt.figure(figsize=(5, 5))
    plt.pie(
        num_images,
        labels=categories,
        autopct="%1.1f%%",
        startangle=140,
        colors=custom_colors,
    )
    plt.title("Distribution of Images by Category")
    plt.axis("equal")
    plt.show()
else:
    print("No images found in the dataset for both Fire and No Fire categories.")
    exit(1)


# Load ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
base_model.layers[0].trainable = False  # Freeze the base model

# Build the classification model on top of ResNet50
model = Sequential()
model.add(base_model)
model.add(Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# ImageDataGenerator for data augmentation and preprocessing
image_size = 416
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode="categorical",
)

validation_generator = data_generator.flow_from_directory(
    val_data_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode="categorical",
)

testing_generator = data_generator.flow_from_directory(
    test_data_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode="categorical",
)

# Early stopping and checkpointing
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", save_best_only=True
)


# Train the model
epochs = 1  # You can adjust the epochs based on performance
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint],
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(testing_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make predictions on the validation set
predictions = model.predict(validation_generator)
