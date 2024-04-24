# Draft-Beer-Classification

## 1. Introduction

Let me introduice you to this ML Engineering project which consist to classify images of Belgian draft beers. 
There are 5 kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel and Westvleteren 12. The classes are numbered 0 to 4 respectively. 


## 2. The Data

The data used for training and local evaluation are placed in the `data` folder in the base folder.
The data has already been split into training data and evaluation data. They are respectively in the `train` folder and `eval` folder. In each you will find four folders which represent the beers you'll need to classify. 
As said, there are five kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel and Westvleteren 12. The classes are numbered 0 to 4 respectively. These class numbers are necessary to create a correct classifier. In data.py, you'll find tthe code to load the images into NumPy arrays and label corresponding to the images.


Tree files : 
* Trainer
  * data.py : contains functions to create_data_with_labels, to collect_paths_to_files, to preprocess_image, to create a training data generator with data augmentation, to check if images need to be normalized and to check whether or not the the files/classes are balanced. 
  * final_task.py
  * model.py : contains the CNN model
  * 



## 3. The Model
## 4. Deploying the Model
## 5. Checking Deployed Model
