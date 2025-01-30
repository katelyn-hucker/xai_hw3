# AIPI 590 Explainable AI: Homework 3 "Explainable Techniques"
### Katelyn Hucker (kh509)

### About: 
In this homework, I used the YOLO Classification blaxkbox model with the following dataset from [ huggingface](https://huggingface.co/datasets/lucabaggi/animal-wildlife) I used the explainable technique "Anchors." I used the Anchors technique so I could see what part of the image prediction mattered to the YOLO model. 

### Steps: 

<u> **Step 0:** </u>

Preprocess the data. This takes the huggingface data and puts it into the YOLO format. 

<u> **Step 1:** </u>

Train the Yolo model with the dataset. Using the colab notebook, you can load the dataset directly from hugging face. However, it has to have the simialr data structure as the dataset defined above. 
The YOLO training process took about 20 mins with Colab GPUs. 

<u> **Step 2:** </u>

Load the model and test it on a different, new image. This lets you upload an image from your loca device and see what the model predicts. 

<u> **Step 3:** </u>

This predicts the image then takes the 'Anchors' library, spefically  the "AnchorImage explainer process" to find most stable and meaningful explanation for the model's prediction. 

### Results and Discussion:

![image](https://github.com/user-attachments/assets/96704992-a97d-4095-89a4-3e6c33ff2066)

I used 'Anchors' as my explainable method, because I wanted to understand how YOLO finds objects, or classifies in this case. This is very useful for my capstone project. For my anchor segmentation method I used the "slic" segmentation method. For efficiency purposes, I limited how much the image was parsed, and it still showed good results.

Strengths
We see above the anchor image highlights most of the bee features are highlighted. Specifically, the legs, antennas, and rear of the bee. However, it is great that the background of the image was completely phased out. This also worked well fairly quickly using the A100 GPU only about 5mins. It was easy to set up with one image/YOLO.

Limitations
I tried the 'quickshift' segmentation method, with different threshold, batches, etc but this took a very long time and would cancel my runtime. I was unable to get it to work with this segmentation method. The different inputs you use can drastically change length of time or results, so it is important the person using the anchor method understands the inputs.

Potential improvements to your approach
I would like to demonstrate this method on other prediction classes, as well as play with the segmentation methods and options. This could enhance how the explainable model registers what YOLO is doing.



