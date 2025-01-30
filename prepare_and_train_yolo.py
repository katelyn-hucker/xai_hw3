import os
import io
import uuid
import ipywidgets as widgets
from IPython.display import display
from datasets import load_dataset
from tqdm import tqdm
import torch
from ultralytics import YOLO
from PIL import Image
import requests
from sklearn.model_selection import train_test_split
def prepare_yolo_dataset():
    """
    Loads a dataset from Hugging Face and organizes it into a YOLO-compatible directory structure.
    Uses an interactive input box in Google Colab for the dataset name.
    """
    dataset_input = widgets.Text(
        value="lucabaggi/animal-wildlife",
        placeholder="Enter Hugging Face dataset name",
        description="Dataset:",
        layout=widgets.Layout(width="50%")
    )
    load_button = widgets.Button(description="Load Dataset")
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            dataset_name = dataset_input.value.strip()

            if not dataset_name:
                print("Please enter a valid dataset name.")
                return

            print(f"Loading dataset: {dataset_name} ...")

            # Load dataset
            try:
                dataset = load_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return

            output_dir = "yolo_dataset"
            os.makedirs(output_dir, exist_ok=True)

            for split in ["train", "test"]:
                split_dir = os.path.join(output_dir, split)
                os.makedirs(split_dir, exist_ok=True)

                def process_example(example):
                    """Processes a single example by saving the image in the correct directory."""
                    label = str(example["label"])
                    class_dir = os.path.join(split_dir, label)
                    os.makedirs(class_dir, exist_ok=True)  # Ensures class folder exists

                    image_path = os.path.join(class_dir, f"{uuid.uuid4().hex}.jpg")
                    example["image"].save(image_path)

                dataset[split].map(process_example, batched=False)  # Faster than looping

            print(f"Dataset successfully prepared in '{output_dir}'.")

    load_button.on_click(on_button_click)
    display(dataset_input, load_button, output)


def load_or_train_model():
    """
    Loads an existing YOLO model if available; otherwise, trains a new model using the specified dataset.
    Uses Colab widgets to allow users to specify the model path and training directory.
    """

    # Widgets for user input
    model_path_input = widgets.Text(
        value="/content/runs/classify/train/weights/best.pt",
        placeholder="Enter the path to your saved model...",
        description="Model Path:",
        layout=widgets.Layout(width="70%"),
    )

    dataset_path_input = widgets.Text(
        value="/content/yolo_dataset",
        placeholder="Enter dataset path...",
        description="Dataset Path:",
        layout=widgets.Layout(width="70%"),
    )

    epochs_input = widgets.IntText(
        value=50, description="Epochs:", layout=widgets.Layout(width="30%")
    )

    img_size_input = widgets.IntText(
        value=224, description="Image Size:", layout=widgets.Layout(width="30%")
    )

    batch_size_input = widgets.IntText(
        value=16, description="Batch Size:", layout=widgets.Layout(width="30%")
    )

    train_button = widgets.Button(
        description="Train/Load Model", button_style="primary"
    )
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()

            # Get user inputs
            model_path = model_path_input.value.strip()
            dataset_path = dataset_path_input.value.strip()
            epochs = epochs_input.value
            img_size = img_size_input.value
            batch_size = batch_size_input.value

            if not model_path or not dataset_path:
                print("Error: Please enter valid paths for the model and dataset.")
                return

            if os.path.exists(model_path):
                print(f"Loading existing model from {model_path} ...")
                model = torch.load(model_path)
                print("Model loaded successfully.")
            else:
                print(f"Model not found at {model_path}. Training a new model...")

                # Load YOLO model
                model = YOLO(
                    "yolov8s-cls.pt"
                )  # Load pretrained YOLOv8 classification model

                # Train the model
                model.train(
                    data=dataset_path, epochs=epochs, imgsz=img_size, batch=batch_size
                )
                print("Training complete.")

                # Validate the model
                print("Validating model...")
                results = model.val(data=dataset_path, split="test")
                print("Validation complete.")
                # Save the trained model to the specified path
                model.save(model_path)
                print(f"Model saved to {model_path}")

    train_button.on_click(on_button_click)

    # Display widgets
    display(
        model_path_input,
        dataset_path_input,
        epochs_input,
        img_size_input,
        batch_size_input,
        train_button,
        output,
    )


# Function to predict the class of an uploaded image
def predict_class():
    """
    Takes an uploaded image, uses the trained YOLO model to predict the class, and outputs the result.
    """
    # Widgets for user input
    model_path_input = widgets.Text(
        value="yolo_trained_model.pt",  # Default model path
        placeholder="Enter the path to your saved model...",
        description="Model Path:",
        layout=widgets.Layout(width="70%"),
    )

    upload_button = widgets.FileUpload(accept="image/*", multiple=False)
    predict_button = widgets.Button(description="Predict Class", button_style="primary")
    output = widgets.Output()

    def on_predict_click(b):
        with output:
            output.clear_output()

            # Ensure a file is uploaded
            if not upload_button.value:
                print("Please upload an image first.")
                return

            # Get uploaded file
            uploaded_file = next(iter(upload_button.value.values()))
            image_bytes = uploaded_file["content"]

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Display uploaded image
            display(image)

            # Load the trained YOLO model
            model_path = model_path_input.value.strip()
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Please train the model first.")
                return

            try:
                # Directly load the trained YOLO model
                model = YOLO(model_path)  # Use YOLO to load the model
            except Exception as e:
                print(f"Error loading model: {e}")
                return

            # Run YOLO model prediction
            results = model.predict(image)

            # Extract class prediction
            if results and len(results) > 0:
                first_result = results[0]

                # Ensure `probs` exists and extract the highest probability class
                if hasattr(first_result, "probs") and hasattr(first_result.probs, "data"):
                    probs_tensor = first_result.probs.data
                    predicted_class_index = torch.argmax(probs_tensor).item()
                    predicted_class_name = first_result.names[predicted_class_index]

                    print(f"Predicted class: {predicted_class_name}")
                else:
                    print("Unable to predict class from the provided results.")
            else:
                print("No results available.")

    # Bind the click event to the button
    predict_button.on_click(on_predict_click)

    # Display widgets
    display(model_path_input, upload_button, predict_button, output)