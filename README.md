# Classifying Fashion-MNIST with Keras using limited data

The goal of the this project is to build a classifier using CNN that is trained using only *one* training observertaion per class. 

## Dataset
![Dataset](/img/data_classes.png "Dataset" =250x)

There are available two subsets of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset for training a CNN:

 1) subset contains 30,000 labeled training examples of the classes 1,2,3,8,9.
 2) subset contains 5 labeled training exmaple from the classes 0,4,5,6,7 (i.e., one per class) and 29992 unlabeled examples.

Furthermore, we have available a test set for all classes containing 1,000 labelled data observation per class. Nevertheless, the classification performance is evaluated only on test set of classes 0,4,5,6,7. 

## Approach
We carried out three task: 

1) Created a baseline. The baseline is a CNN trained on 5 augmented examples of 0,4,5,6,7.

2) Performed transfer learning. We trained a CNN model using data from classes 1,2,3,8,9 and then use the weights from this model to initilize a CNN model that is trained just like the baseline.

3) Performed one-shot learning. We created a siamese network following approach of Koch et. al (2015) with slight modifications. We created 100 correct and 100 incorect pairs for each class; in total 1000 pairs. We trained the siamese network with 70% of this data. The rest of the data were used for validation. The siamese network scores 95% on the verificaiton tasks (i.e., recognizing pairs).

## Results
The results of the models are evaluated using accuracy on the set of classes 0,4,5,6,7. 

| Model  | Accuracy |
| ------------- | ------------- |
| 1. Baseline  | 66.2% |
| 2. Transfer learning  | 66.6% |
| 3. Siamese network  | 45%  |

## File structure
    ├── classes                    
    │   ├── base_model.py          # Class handling a basic CNN model
    │   ├── data_handler.py        # Class handling all data manipulation
    │   ├── nn_network.py          # A base class for all models
    │   └── siamese_nn.py          # Class handeling one-shot learning
    │   └── transfer_learning.py   # Class handeling transfer learning
    └── img                        # Images for documention 
    └── FashionData                # Images for documention 
    │   ├── FashionPDEngDM.npz     # Prepared [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) data set (already splitted into the classes)
    └── Readme.md                    
    └── Run.ipynb                  # The main access point

## Acknowledgement
This project was a part of Data Mining module in [JADS](http://jads.nl) led by Vlado Menkovski and Decebal Mocanu.


