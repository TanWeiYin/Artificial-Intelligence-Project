# TRAFFIC SIGN ClASSIFICATION (TSR) USING DEEP LEARNING 

## A. PROJECT SUMMARY

**Project Title:** Face Mask Detection using Deep Learning

**Team Members:** 
- [insert Member Name]
- [insert Member Name]
- [insert Member Name]
- [insert Member Name]


- [ ] **Objectives:**
- Recognizing traffic signs along the road
- Automatically recognize traffic signs enables us to build “smarter cars”
- “Driver Alert” systems inside cars need to understand the roadway around them to help aid and protect drivers.

**Challenges:** 
- Some images are low resolution, and worse, have poor contrast 

##  B. ABSTRACT 

Last weekend I drove down to Maryland to visit my parents. As I pulled into their driveway I noticed something strange — there was a car I didn’t recognize sitting in my dad’s parking spot.

I parked my car, grabbed my bags out of the trunk, and before I could even get through the front door, my dad came out, excited and enlivened, exclaiming that he had just gotten back from the car dealership and traded in his old car for a brand new 2020 Honda Accord.

Most everyone enjoys getting a new car, but for my dad, who puts a lot of miles on his car each year for work, getting a new car is an especially big deal.

My dad wanted the family to go for a drive and check out the car, so my dad, my mother, and I climbed into the vehicle, the “new car scent” hitting you like bad cologne that you’re ashamed to admit that you like.

As we drove down the road my mother noticed that the speed limit was automatically showing up on the car’s dashboard — how was that happening?

The answer?

Traffic sign recognition.

In the 2020 Honda Accord models, a front camera sensor is mounted to the interior of the windshield behind the rearview mirror.

That camera polls frames, looks for signs along the road, and then classifies them.

The recognized traffic sign is then shown on the LCD dashboard as a reminder to the driver.

It’s admittedly a pretty neat feature and the rest of the drive quickly turned from a vehicle test drive into a lecture on how computer vision and deep learning algorithms are used to recognize traffic signs (I’m not sure my parents wanted that lecture but they got it anyway).

When I returned from visiting my parents I decided it would be fun (and educational) to write a tutorial on traffic sign recognition — you can use this code as a starting point for your own traffic sign recognition projects.


![Coding](https://pyimagesearch.com/wp-content/uploads/2019/11/traffic_sign_recognition_header.jpg)

Figure 1 shows the AI output of detecting traffic sign on the road and classifying them .


## C.  DATASET

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used for demostrating TSR.

I’ll then show you how to implement a Python script to train a traffic sign detector on our dataset using Keras and TensorFlow.

We’ll use this Python script to train a traffic sign detector and review the results.

Given the trained traffic sign detector, we’ll proceed to implement two more additional Python scripts used to:

- Traffic Sign Detection: Detect all the signs from a given video frame
- Traffic Sign Recognition: Recognize all the detected signs

We’ll wrap up the post by looking at the results of applying our traffic sign detector.


There is two-phase traffic signs detector as shown in Figure 2:

![Figure 2](https://pyimagesearch.com/wp-content/uploads/2019/11/traffic_sign_classification_phases.jpg)

Figure 2: Phases and individual steps for building a traffic signs detector with computer vision and deep learning 


Our taffic signs detection dataset as shown in Figure 3:

![Figure 3](https://pyimagesearch.com/wp-content/uploads/2019/11/traffic_sign_classification_dataset.jpg)

Figure 3: The German Traffic Sign Recognition Benchmark (GTSRB) dataset will be used for traffic sign classification with Keras and deep learning.

The dataset we’ll be using here today was created by Kaggle user, Mykola

The GTSRB dataset consists of 43 traffic sign classes and nearly 50,000 images.

Our goal is to train a custom deep learning model to detect whether there is a traffic sign in the image and what sign is it.

Source of dataset: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
-$ tree --dirsfirst --filelimit 10
-.
-├── examples [25 entries]
-├── gtsrb-german-traffic-sign
-│   ├── Meta [43 entries]
-│   ├── Test [12631 entries]
-│   ├── Train [43 entries]
-│   ├── meta-1 [43 entries]
-│   ├── test-1 [12631 entries]
-│   ├── train-1 [43 entries]
-│   ├── Meta.csv
-│   ├── Test.csv
-│   └── Train.csv
-├── output
-│   ├── trafficsignnet.model
-│   │   ├── assets
-│   │   ├── variables
-│   │   │   ├── variables.data-00000-of-00002
-│   │   │   ├── variables.data-00001-of-00002
-│   │   │   └── variables.index
-│   │   └── saved_model.pb
-│   └── plot.png
-├── pyimagesearch
-│   ├── __init__.py
-│   └── trafficsignnet.py
-├── train.py
-├── signnames.csv
-└── predict.py
-13 directories, 13 files


Our project contains three main directories and one Python module:

- gtsrb-german-traffic-sign/ : Our GTSRB dataset.
- output/ : Contains our output model and training history plot generated by train.py .
- examples/ : Contains a random sample of 25 annotated images generated by predict.py .
- pyimagesearch : A module that comprises our TrafficSignNet CNN.


## E.  Implementing TrafficSignNet, our CNN traffic sign classifier

![Figure 4](https://pyimagesearch.com/wp-content/uploads/2019/11/traffic_sign_recognition_architecture.png)

Figure 5: The Keras deep learning framework is used to build a Convolutional Neural Network (CNN) for traffic sign classification.

Let’s go ahead and implement a Convolutional Neural Network to classify and recognize traffic signs.

Name this classifier TrafficSignNet — open up the trafficsignnet.py file in your project directory and then insert the following code:

-# import the necessary packages
-from tensorflow.keras.models import Sequential
-from tensorflow.keras.layers import BatchNormalization
-from tensorflow.keras.layers import Conv2D
-from tensorflow.keras.layers import MaxPooling2D
-from tensorflow.keras.layers import Activation
-from tensorflow.keras.layers import Flatten
-from tensorflow.keras.layers import Dropout
-from tensorflow.keras.layers import Dense
-class TrafficSignNet:
-	@staticmethod
-	def build(width, height, depth, classes):
-		# initialize the model along with the input shape to be
-		# "channels last" and the channels dimension itself
-		model = Sequential()
-		inputShape = (height, width, depth)
-		chanDim = -1

 define our CONV => RELU => BN => POOL  layer set:

- 		# CONV => RELU => BN => POOL
-		model.add(Conv2D(8, (5, 5), padding="same",
-			input_shape=inputShape))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization(axis=chanDim))
-		model.add(MaxPooling2D(pool_size=(2, 2)))

From there we define two sets of (CONV => RELU => CONV => RELU) * 2 => POOL layers:

-		# first set of (CONV => RELU => CONV => RELU) * 2 => POOL
-		model.add(Conv2D(16, (3, 3), padding="same"))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization(axis=chanDim))
-		model.add(Conv2D(16, (3, 3), padding="same"))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization(axis=chanDim))
-		model.add(MaxPooling2D(pool_size=(2, 2)))
-		# second set of (CONV => RELU => CONV => RELU) * 2 => POOL
-		model.add(Conv2D(32, (3, 3), padding="same"))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization(axis=chanDim))
-		model.add(Conv2D(32, (3, 3), padding="same"))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization(axis=chanDim))
-		model.add(MaxPooling2D(pool_size=(2, 2)))

The head of our network consists of two sets of fully connected layers and a softmax classifier:

-		# first set of FC => RELU layers
-		model.add(Flatten())
-		model.add(Dense(128))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization())
-		model.add(Dropout(0.5))
-		# second set of FC => RELU layers
-		model.add(Flatten())
-		model.add(Dense(128))
-		model.add(Activation("relu"))
-		model.add(BatchNormalization())
-		model.add(Dropout(0.5))
-		# softmax classifier
-		model.add(Dense(classes))
-		model.add(Activation("softmax"))
-		# return the constructed network architecture
-		return model

Dropout is applied as a form of regularization which aims to prevent overfitting. The result is often a more generalizable model.

## F.  Implementing our training script

Now that our TrafficSignNet architecture has been implemented, let’s create our Python training script that will be responsible for:

-Loading our training and testing split from the GTSRB dataset
-Preprocessing the images
-Training our model
-Evaluating our model’s accuracy
-Serializing the model to disk so we can later use it to make predictions on new traffic sign data

open up the train.py file in your project directory and add the following code:

-# set the matplotlib backend so figures can be saved in the background
-import matplotlib
-matplotlib.use("Agg")
-# import the necessary packages
-from pyimagesearch.trafficsignnet import TrafficSignNet
-from tensorflow.keras.preprocessing.image import ImageDataGenerator
-from tensorflow.keras.optimizers import Adam
-from tensorflow.keras.utils import to_categorical
-from sklearn.metrics import classification_report
-from skimage import transform
-from skimage import exposure
-from skimage import io
-import matplotlib.pyplot as plt
-import numpy as np
-import argparse
-import random
-import os

define a function to load our data from disk:

-def load_split(basePath, csvPath):
-	# initialize the list of data and labels
-	data = []
-	labels = []
-	# load the contents of the CSV file, remove the first line (since
-	# it contains the CSV header), and shuffle the rows (otherwise
-	# all examples of a particular class will be in sequential order)
-	rows = open(csvPath).read().strip().split("\n")[1:]
-	random.shuffle(rows)

loop over the rows  now and extract + preprocess the data that we need:

-	# loop over the rows of the CSV file
-	for (i, row) in enumerate(rows):
-		# check to see if we should show a status update
-		if i > 0 and i % 1000 == 0:
-			print("[INFO] processed {} total images".format(i))
-		# split the row into components and then grab the class ID
-		# and image path
-		(label, imagePath) = row.strip().split(",")[-2:]
-		# derive the full path to the image file and load it
-		imagePath = os.path.sep.join([basePath, imagePath])
-		image = io.imread(imagePath)

## G.  Training TrafficSignNet on the traffic sign dataset

Open up a terminal and execute the following command:

-$ python train.py --dataset gtsrb-german-traffic-sign \
-	--model output/trafficsignnet.model --plot output/plot.png
-[INFO] loading training and testing data...
-[INFO] compiling model...
-[INFO] training network...
-Epoch 1/30
-612/612 [==============================] - 49s 81ms/step - loss: 2.6584 - accuracy: 0.2951 - val_loss: 2.1152 - val_accuracy: 0.3513
-Epoch 2/30
-612/612 [==============================] - 47s 77ms/step - loss: 1.3989 - accuracy: 0.5558 - val_loss: 0.7909 - val_accuracy: 0.7417
-Epoch 3/30
-612/612 [==============================] - 48s 78ms/step - loss: 0.9402 - accuracy: 0.6989 - val_loss: 0.5147 - val_accuracy: 0.8302
-Epoch 4/30
-612/612 [==============================] - 47s 76ms/step - loss: 0.6940 - accuracy: 0.7759 - val_loss: 0.4559 - val_accuracy: 0.8515
-Epoch 5/30
-612/612 [==============================] - 47s 76ms/step - loss: 0.5521 - accuracy: 0.8219 - val_loss: 0.3004 - val_accuracy: 0.9055
-...
-Epoch 26/30
-612/612 [==============================] - 46s 75ms/step - loss: 0.1213 - accuracy: 0.9627 - val_loss: 0.7386 - val_accuracy: 0.8274
-Epoch 27/30
-612/612 [==============================] - 46s 75ms/step - loss: 0.1175 - accuracy: 0.9633 - val_loss: 0.1931 - val_accuracy: 0.9505
-Epoch 28/30
-612/612 [==============================] - 46s 75ms/step - loss: 0.1101 - accuracy: 0.9664 - val_loss: 0.1553 - val_accuracy: 0.9575
-Epoch 29/30
-612/612 [==============================] - 46s 76ms/step - loss: 0.1098 - accuracy: 0.9662 - val_loss: 0.1642 - val_accuracy: 0.9581
-Epoch 30/30
-612/612 [==============================] - 47s 76ms/step - loss: 0.1063 - accuracy: 0.9684 - val_loss: 0.1778 - val_accuracy: 0.9495
-[INFO] evaluating network...
|      |    precision    | recall| f1-score | support |
|------|-----------------|-------|----------|---------|
|Speed limit (20km/h)|0.94|0.98|0.96|60|
|Speed limit (30km/h)|0.96|0.97|0.97|720|
|Speed limit (50km/h)|0.95|0.98|0.96|750|
|Speed limit (60km/h)|0.98|0.92|0.95|450|
|Speed limit (70km/h)|0.98|0.96|0.97|660|
|Speed limit (80km/h)|0.92|0.93|0.93|630|
|End of speed limit (80km/h)|0.96|0.87|0.91|150|
|Speed limit (100km/h)|0.93|0.94|0.93|450|
|Speed limit (120km/h)|0.90|0.99|0.94|450|
|No passing|1.00|0.97|0.98|480|
|No passing veh over 3.5 tons|1.00|0.96|0.98|660|
|Right-of-way at intersection|0.95|0.93|0.94|420|
|Priority road|0.99|0.99|0.99|690|
|Yield|0.98|0.99|0.99|720|
|Stop|1.00|1.00|1.00|270|
|No vehicles|0.99|0.90|0.95|210|
|Veh > 3.5 tons prohibited|0.97|0.99|0.98|150|
|No entry|1.00|0.94|0.97|360|
|General caution|0.98|0.77|0.86|390|
|Dangerous curve left|0.75|0.60|0.660|
|Dangerous curve right|0.69|1.00|0.81|90|
|Double curve|0.76|0.80|0.78|90|
|Bumpy road|0.99|0.78|0.87|120|
|Slippery road|0.66|0.99|0.79|150|
|Road narrows on the right|0.80|0.97|0.87|90|
|Road work|0.94|0.98|0.96|480|
|Traffic signals|0.87|0.95| 0.91|180|
|Pedestrians|0.46|0.55|0.50|60|
|Children crossing|0.93|0.94|0.94|150|
|Bicycles crossing|0.92|0.86|0.89|90|
|Beware of ice/snow|0.88|0.75|0.81|150|
|Wild animals crossing|0.98|0.95|0.96|270|
|End speed + passing limits|0.98|0.98|0.98|60|
|Turn right ahead|0.97|1.00|0.98|210|
|Turn left ahead|0.98|1.00|0.99|120|
|Ahead only|0.99|0.97|0.98|390|
|Go straight or right|1.00|1.00|1.00|120|
|Go straight or left|0.92|1.00|0.96|60|
|Keep right|0.99|1.00|0.99|690|
|Keep left|0.97|0.96|0.96|90|
|Roundabout mandatory|0.90|0.99|0.94|90|
|End of no passing|0.90|1.00|0.94|60|
|End no passing veh > 3.5 tons|0.91|0.89|0.90|90|
|accuracy| | |0.95|12630|
|macro avg|0.92|0.93|0.92|12630|
|weighted avg|0.95|0.95|0.95|12630|
-[INFO] serializing network to 'output/trafficsignnet.model'...


![Figure 5](https://pyimagesearch.com/wp-content/uploads/2019/11/plot.png)

Figure 4: Keras and deep learning is used to train a traffic sign classifier.


## H.  RESULT AND CONCLUSION

In this project, you learned how to create a Traffic Signs Classification using OpenCV, Keras/TensorFlow, and Deep Learning

Train and classify Traffic Signs using Convolutional neural networks This will be done using  OPENCV in real time using a simple webcam . CNNs have been gaining popularity in the past couple of years due to their ability to generalize and classify the data with high accuracy.

By the end of the video  I be will sharing information that will help you classify your own data set. Info such as  how long does it take to train and how much data of each class is required to have a good classification model. 

[![Figure6](https://www.youtube.com/watch?v=SWaYRyi0TTs&ab_channel=Murtaza%27sWorkshop-RoboticsandAI "Figure6")

Figure 5: Traffic Signs Classification in real-time video streams


## I.  PROJECT PRESENTATION 

In this video, I am showing you the tutorial of Traffic🚦 Sings Classification using CNN and We made a WebApp using Flask.
What I have Covered in this video

- Dataset Handling
- Image Processing 
- CNN Fitting in Data
- Live implementation and testing
- Implement Deep Learning in Real-time Application

[![demo](https://www.youtube.com/watch?v=qahpZkPlTRM&ab_channel=MachineLearningHub "demo")




