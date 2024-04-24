# 🍺Draft-Beer-Classification🍺

## 1️⃣ Introduction

Let me introduce you to this ML Engineering project, which aims to classify images of Belgian draft beers. 
There are five kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel, and Westvleteren 12. The classes are numbered 0 to 4, respectively.

## 2️⃣ The Data

The data used for training and local evaluation are placed in the `data` folder in the base folder.
The data has already been split into training data and evaluation data. They are respectively in the `train` folder and `eval` folder. In each there are four folders which represent the beers to classify. 
As said, there are five kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel and Westvleteren 12. The classes are numbered 0 to 4 respectively. These class numbers are necessary to create a correct classifier. In data.py, there is code to load the images into NumPy arrays and label corresponding to the images. 
The number of images are balanced as follow : 

* Classe'0': Total of train images 108
* Classe'1': Total of train images 91
* Classe'2': Total of train images 83
* Classe'3': Total of train images 114
* Classe'4': Total of train images 104

Three key files:

* data.py: Contains functions to load and label data, preprocess images, create a training data generator with data augmentation, normalize images if necessary, and check the balance of files/classes.
* final_task.py: Contains functions to prepare an image tensor for prediction, prepare batches of images, export the model, and get training data from the specified folders.
* model.py: Contains the CNN model definition.
* params.py: Contains parameters for the model deployed to Google Cloud Vertex AI.
* task.py: Manages the training process using data from the train and eval folders as input for the model.py.

## 3️⃣ The Model

The model is a Convolutional Neural Network (CNN) achieving 90% accuracy. The accuracy was further improved to 92% by employing transfer learning with the renowned pretrained VGG16 model.

## 4️⃣ Deploying the Model

The model is designed to be deployed on Google Cloud as an API that can receive new images of draft beers and return predictions. The deployment process is initiated by running a command in the command line, as outlined in final_task.py. To export the trained model and train it on images from the train and eval folders, the following command can be executed (after completing the model.py code):

```bash
python -m trainer.final_task
```
After executing this command, a folder named output will be created in the root directory of this repository. This folder contains the model ready for deployment to Google Cloud Vertex AI.

## 5️⃣ Checking Deployed Model

To check if the deployed model works correctly, the following commands can be executed :

```bash
ENDPOINT_ID=<your_endpoint_id>
PROJECT_ID=<your_project_id>

gcloud ai endpoints predict $ENDPOINT_ID \
    --project=$PROJECT_ID  \
    --region=europe-west1 \
    --json-request=check_deployed_model/test.json
```

Ensure to retrieve a prediction from the gcloud command. If there are errors, part of code must be resolve. The output of the command should look similar to this (though the numbers may vary):

```
CLASSES  PROBABILITIES
1        [0.004145291168242693, 0.9800060987472534, 0.004468264523893595, 0.007732450030744076, 0.0036478929687291384]
```

Enjoy reading the code🤓
