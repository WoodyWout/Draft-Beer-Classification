# Draft-Beer-Classification

## 1. Introduction

Let me introduice you to this ML Engineering project which consist to classify images of Belgian draft beers. 
There are 5 kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel and Westvleteren 12. The classes are numbered 0 to 4 respectively. 


## 2. The Data

The data used for training and local evaluation are placed in the `data` folder in the base folder.
The data has already been split into training data and evaluation data. They are respectively in the `train` folder and `eval` folder. In each you will find four folders which represent the beers you'll need to classify. 
As said, there are five kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel and Westvleteren 12. The classes are numbered 0 to 4 respectively. These class numbers are necessary to create a correct classifier. In data.py, you'll find tthe code to load the images into NumPy arrays and label corresponding to the images. The number of images are balanced as follow : 

* Classe'0': Total of train images 108
* Classe'1': Total of train images 91
* Classe'2': Total of train images 83
* Classe'3': Total of train images 114
* Classe'4': Total of train images 104

Tree files : 
* Trainer
  * data.py : contains functions to create_data_with_labels, to collect_paths_to_files, to preprocess_image, to create a training data generator with data augmentation, to check if images need to be normalized and to check whether or not the the files/classes are balanced. 
  * final_task.py : containes functions to prepare an image tensor for prediction, to prepare a batch of images for prediction, to prepare model for strings representing image data and export it, to get the training data from the training folder and the eval folder
  * model.py : contains the CNN model.
  * params.py : contains the parameters of the model deployed to Google Cloud Vertex AI.
  * task.py : contains the function that gets the training data from the training folder, the evaluation data from the eval folder and trains the solution input from the model.py file with it. 

## 3. The Model

The model is a Convolutional Neural Network (CNN) attaining a 90% accuracy. I've also created a model that improves accuracy to 92% by employing transfer learning with the famous pretrained model VGG16.

## 4. Deploying the Model

Once the model working, the idea was to deploy the model to Google Cloud to turn it into an API that can receive new images of draft beers and returns its prediction them. 
The code for this is written in the `final_task.py` file. To deploy the model, you have to run a few commands in your command line.

To export the trained model and to train the model on the images in the `train` and `eval` folder, you have to execute the following command (only do this once you've completed coding the `model.py` file):

```bash
python -m trainer.final_task
```

## 5. Checking Deployed Model


