# AIPI 590 Explainable AI: Homework 3 "Explainable Techniques"
### Katelyn Hucker (kh509)

### About: 
In this homework, I used the YOLO Classification blaxkbox model with the following dataset from [rom huggingface](https://huggingface.co/datasets/lucabaggi/animal-wildlife) I used the explainable technique "Anchors." I used the Anchors technique so I could see what part of the image prediction mattered to the YOLO model. 

### Steps: 

<u>**Step 0:**</u>

Preprocess the data. This takes the huggingface data and puts it into the YOLO format. 

<u>**Step 1:**</u>

Train the Yolo model with the dataset. Using the colab notebook, you can load the dataset directly from hugging face. However, it has to have the simialr data structure as the dataset defined above. 
The YOLO training process took about 20 mins with Colab GPUs. 

<u>**Step 2:**</u>

Load the model and test it on a different, new image. This lets you upload an image from your loca device and see what the model predicts. 

<u>**Step 3:**</u>

This predicts the image then takes the 'Anchors' library, spefically  the "AnchorImage explainer process" to find most stable and meaningful explanation for the model's prediction. 


