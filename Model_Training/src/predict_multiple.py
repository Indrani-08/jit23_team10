# ============================================
# Project : Nail Nutrition
# Module  : Multiple Nail Image Prediction
# ============================================

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================
# Configuration
# ============================================

MODEL_PATH = "models/final_nail_model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Healthy",
    "Iron Deficiency",
    "Vitamin B12 Deficiency",
    "Vitamin D Deficiency"
]


# ============================================
# Load Trained Model
# ============================================

print("=" * 60)
print("LOADING NAIL NUTRITION MODEL")
print("=" * 60)

model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("Model loaded successfully!")


# ============================================
# Preprocess One Image
# ============================================

def preprocess_image(image_path):

    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================
# Predict One Image
# ============================================

def predict_single_image(image_path):

    processed_image = preprocess_image(image_path)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# ============================================
# Predict Multiple Images
# ============================================

def predict_multiple_images(image_paths):

    results = []

    print("\n" + "=" * 60)
    print("ANALYZING NAIL IMAGES")
    print("=" * 60)

    for image_path in image_paths:

        if not os.path.exists(image_path):

            print("\nImage not found:")
            print(image_path)

            continue

        predicted_class, confidence = predict_single_image(
            image_path
        )

        result = {
            "image": image_path,
            "prediction": predicted_class,
            "confidence": confidence
        }

        results.append(result)

        print("\nImage:", os.path.basename(image_path))
        print("Prediction:", predicted_class)
        print(f"Confidence: {confidence:.2f}%")


    return results


# ============================================
# Combine Results
# ============================================

def combine_results(results):

    if not results:

        return []

    detected_conditions = {}

    for result in results:

        condition = result["prediction"]
        confidence = result["confidence"]

        if condition == "Healthy":
            continue

        if condition not in detected_conditions:

            detected_conditions[condition] = []

        detected_conditions[condition].append(
            confidence
        )

    final_results = []

    for condition, confidences in detected_conditions.items():

        average_confidence = np.mean(confidences)

        final_results.append({
            "condition": condition,
            "confidence": average_confidence,
            "image_count": len(confidences)
        })

    final_results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return final_results


# ============================================
# Main Program
# ============================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("NAIL NUTRITION - MULTIPLE IMAGE ANALYSIS")
    print("=" * 60)

    print("\nEnter the number of nail images:")
    
    try:

        number_of_images = int(
            input("Number of images: ")
        )

    except ValueError:

        print("Please enter a valid number.")
        exit()


    image_paths = []


    for i in range(number_of_images):

        print(f"\nEnter path for image {i + 1}:")

        image_path = input("> ").strip().strip('"')

        if os.path.exists(image_path):

            image_paths.append(image_path)

        else:

            print("Image not found:")
            print(image_path)


    if not image_paths:

        print("\nNo valid images were provided.")
        exit()


    # ========================================
    # Run Predictions
    # ========================================

    results = predict_multiple_images(
        image_paths
    )


    # ========================================
    # Combine Predictions
    # ========================================

    final_results = combine_results(
        results
    )


    # ========================================
    # Display Final Result
    # ========================================

    print("\n" + "=" * 60)
    print("FINAL PERSON-LEVEL RESULT")
    print("=" * 60)


    if not final_results:

        print("\nNo deficiency was detected.")
        print("All analyzed images were classified as Healthy.")

    else:

        print("\nPossible conditions detected:\n")

        for result in final_results:

            print(
                f"{result['condition']} : "
                f"{result['confidence']:.2f}% "
                f"({result['image_count']} image(s))"
            )


    print("\nAnalysis completed successfully!")