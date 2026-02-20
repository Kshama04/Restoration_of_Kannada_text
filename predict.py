"""
Kannada Character Recognition - Prediction Script
Usage: python predict.py --image your_image.jpg
"""

import numpy as np
import cv2
import sys
import os

KANNADA_MAPPING = {
    0: 'ಅ', 1: 'ಆ', 2: 'ಇ', 3: 'ಈ', 4: 'ಉ',
    5: 'ಊ', 6: 'ಋ', 7: 'ಎ', 8: 'ಏ', 9: 'ಐ',
    10: 'ಒ', 11: 'ಓ', 12: 'ಔ', 13: 'ಕ', 14: 'ಖ',
    15: 'ಗ', 16: 'ಘ', 17: 'ಙ', 18: 'ಚ', 19: 'ಛ',
    20: 'ಜ', 21: 'ಝ', 22: 'ಞ', 23: 'ಟ', 24: 'ಠ',
    25: 'ಡ', 26: 'ಢ', 27: 'ಣ', 28: 'ತ', 29: 'ಥ',
    30: 'ದ', 31: 'ಧ', 32: 'ನ', 33: 'ಪ', 34: 'ಫ',
    35: 'ಬ', 36: 'ಭ', 37: 'ಮ', 38: 'ಯ', 39: 'ರ',
    40: 'ಲ', 41: 'ವ', 42: 'ಶ', 43: 'ಷ', 44: 'ಸ',
    45: 'ಹ', 46: 'ಳ'
}

MODEL_PATH = 'final_model.keras'

def load_model():
    from tensorflow.keras.models import load_model as keras_load
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: '{MODEL_PATH}'")
        sys.exit(1)
    print(f"✅ Loading model from '{MODEL_PATH}'...")
    model = keras_load(MODEL_PATH)
    # Print what the model actually expects
    print(f"✅ Model loaded! Input shape: {model.input_shape}")
    return model

def preprocess_image(image_path, input_shape):
    # input_shape is like (None, H, W, C)
    h = input_shape[1]
    w = input_shape[2]
    c = input_shape[3]

    if c == 1:
        # Grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read image: '{image_path}'")
            sys.exit(1)
        img = cv2.resize(img, (w, h))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 1)
    else:
        # Colour (3 channels)
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: '{image_path}'")
            sys.exit(1)
        img = cv2.resize(img, (w, h))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)   # (1, H, W, 3)

    return img

def predict(model, image_path):
    img = preprocess_image(image_path, model.input_shape)
    prediction = model.predict(img, verbose=0)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100
    character = KANNADA_MAPPING.get(class_idx, "Unknown")
    return character, confidence, class_idx

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  type=str, help="Path to image file")
    parser.add_argument('--folder', type=str, help="Path to folder of images")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Usage:")
        print("  python predict.py --image your_image.jpg")
        print("  python predict.py --folder your_folder/")
        sys.exit(0)

    model = load_model()

    if args.image:
        char, conf, idx = predict(model, args.image)
        print(f"\nImage     : {args.image}")
        print(f"Predicted : {char}  (class index: {idx})")
        print(f"Confidence: {conf:.2f}%")

    if args.folder:
        supported = ('.jpg', '.jpeg', '.png')
        files = [f for f in os.listdir(args.folder) if f.lower().endswith(supported)]
        print(f"\nFound {len(files)} image(s)\n")
        print(f"{'Image':<35} {'Predicted':<12} {'Confidence'}")
        print("-" * 60)
        for fname in sorted(files):
            path = os.path.join(args.folder, fname)
            char, conf, _ = predict(model, path)
            print(f"{fname:<35} {char:<12} {conf:.1f}%")
