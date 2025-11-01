import joblib
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor
from scripts.preprocess import preprocess_image
import gradio as gr
import tempfile
import numpy as np
from pyngrok import ngrok
# Load the ViT processor and model
proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").eval()

# Load the classifier
clf_bundle = joblib.load("outputs/classifier_final.joblib")
clf = clf_bundle['model']
class_names = clf_bundle.get('class_names') or clf_bundle.get('classes') or clf_bundle.get('present_labels')

def predict(image: Image.Image):
    """
    Predicts the emotion from an input PIL image.
    """
    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    # Preprocess using existing function
    processed_image = preprocess_image(tmp_path)

    if processed_image is None:
        return "No face detected"

    # Convert processed image to PIL format
    pil_image = Image.fromarray(processed_image.astype('uint8'))

    # Prepare input for ViT
    inputs = proc(images=pil_image, return_tensors="pt")

    # Extract ViT features
    with torch.no_grad():
        outputs = vit(**inputs)
        feat = outputs.pooler_output.numpy()

    # Predict with SVM
    pred_idx = clf.predict(feat)[0]
    label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    
    return label

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Thermal Image"),
    outputs="text",
    title="Thermal Emotion Recognition",
    description="Upload a thermal image to predict the emotion."
)

if __name__ == "__main__":
    # Launch Gradio app
    iface.launch(share=False)  # 'share=True' creates a public link if needed
