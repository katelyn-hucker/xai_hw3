def explain_prediction():
    """
    Creates an interactive widget setup for uploading an image, running YOLO prediction,
    and generating AnchorImage explanations in Google Colab.
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import io
    import os
    from PIL import Image
    import torch
    import numpy as np
    from alibi.explainers import AnchorImage
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    from tqdm.notebook import tqdm
    import time
    
    # Create widgets
    model_path_input = widgets.Text(
        value="yolo_trained_model.pt",
        description="Model Path:",
        layout=widgets.Layout(width="70%")
    )
    upload_button = widgets.FileUpload(accept="image/*", multiple=False)
    explain_button = widgets.Button(description="Generate Explanation", button_style="primary")
    progress = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        orientation='horizontal'
    )
    status_label = widgets.Label('Ready...')
    output = widgets.Output()
    
    def preprocess_image(img, img_size=320):  # Reduced size for faster processing
        """Preprocess image for model input and explanation"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((img_size, img_size))  # Reduced image size
        img_array = np.array(img)
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        return img_array

    
    def on_explain_click(b):
        with output:
            output.clear_output()
            display(progress, status_label)
            
            if not upload_button.value:
                status_label.value = "Please upload an image first."
                return
                
            try:
                # Update progress
                progress.value = 10
                status_label.value = "Loading image..."
                
                uploaded_file = next(iter(upload_button.value.values()))
                image_bytes = uploaded_file["content"]
                image = Image.open(io.BytesIO(image_bytes))
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis("off")
                plt.draw()
                plt.pause(0.1)
                
                # Update progress
                progress.value = 20
                status_label.value = "Loading model..."
                
                model_path = model_path_input.value.strip()
                if not os.path.exists(model_path):
                    status_label.value = f"Model not found at {model_path}"
                    return
                    
                model = YOLO(model_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                # Update progress
                progress.value = 30
                status_label.value = "Preprocessing image..."
                
                input_image = preprocess_image(image)
                
                def predict_fn(images):
                    predictions = []
                    for img in images:
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        result = model(img, verbose=False)[0]
                        if hasattr(result, "probs") and hasattr(result.probs, "data"):
                            pred = torch.argmax(result.probs.data).item()
                        else:
                            pred = 0
                        predictions.append(pred)
                    return np.array(predictions)
                
                # Update progress
                progress.value = 40
                status_label.value = "Running initial prediction..."
                
                results = model(input_image, verbose=False)
                if results and len(results) > 0:
                    first_result = results[0]
                    if hasattr(first_result, "probs"):
                        probs_tensor = first_result.probs.data
                        predicted_class_index = torch.argmax(probs_tensor).item()
                        predicted_class_name = first_result.names[predicted_class_index]
                        
                        print(f"Generating explanation for predicted class: {predicted_class_name}")
                        
                        # Update progress
                        progress.value = 50
                        status_label.value = "Creating explainer..."
                        
                        # Create explainer with faster settings
                        explainer = AnchorImage(
                            predict_fn,
                            input_image.shape,
                            segmentation_fn="slic",
                            segmentation_kwargs={
                                'n_segments': 50,  # Adjust for faster processing
                                'compactness': 10,  # Trade-off between segmentation quality and speed
                            }
                        )
                        
                        # Update progress
                        progress.value = 60
                        status_label.value = "Generating explanation (this may take a few minutes)..."
                        
                        # Normalize image
                        input_image_normalized = input_image.astype(np.float32) / 255.0
                        
                        # Generate explanation with reduced parameters for speed
                        explanation = explainer.explain(
                            input_image_normalized,
                            threshold=0.75,  # Reduced threshold for faster convergence
                            coverage_samples=10,  # Fewer samples to speed up computation
                            batch_size=3,  # Smaller batch size for less memory usage
                            delta=0.2  # Kept for faster convergence
                        )
                        # Update progress
                        progress.value = 90
                        status_label.value = "Plotting results..."
                        
                        # Plot explanation
                        plt.subplot(1, 2, 2)
                        plt.imshow(input_image)
                        plt.imshow(explanation.anchor, cmap='gray', alpha=0.5)
                        plt.title(f"Anchor Explanation\nPrediction: {predicted_class_name}")
                        plt.axis("off")
                        
                        print("\nExplanation Details:")
                        print(f"Anchor stability: {explanation.precision:.2f}")
                        print(f"Coverage: {explanation.coverage:.2f}")
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # Complete
                        progress.value = 100
                        status_label.value = "Done!"
                        
                    else:
                        status_label.value = "Unable to get class probabilities from model results."
                else:
                    status_label.value = "No prediction results available."
                    
            except Exception as e:
                status_label.value = f"Error: {str(e)}"
                import traceback
                traceback.print_exc()
    
    # Bind click event
    explain_button.on_click(on_explain_click)
    
    # Display widgets
    display(model_path_input, upload_button, explain_button, output)