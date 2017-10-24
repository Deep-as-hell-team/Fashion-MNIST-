import heapq
import numpy as np
 
#A dictionary of our classes and their label
label = {0:'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 
             5: 'Sandal', 6: 'Shirt', 7:'Sneaker', 8: 'Bag', 9:'Ankle Boot'
          }
    
    
def accuracy_per_class(model, x_data, y_data , level, label = label):
    '''
    A function that returns the accuracy per class based on the level of the probabilities.
    
    Parameters
    ----------
    model: A trained Keras Sequential model
    
    x_data: array
        The array of images
        
    y_data: array
        The hot encoded sequence of the labels
        
    level: int
        The level of the probabilities (Ask Mimis to tell you more about)
        
    label: dict
        A dictionary for the different classes
        
    ---------------------------------------------------------------------------
    Returns:
         The accuracy per class
    '''
    if level>9:
        raise ValueError('The level of should be up to the number of classes')

    #One hot to integer
    one_hot_to_int = np.argmax(y_data, axis = 1)
    #The number of the classes
    unique_classes = set(one_hot_to_int)
    #The prediction probabilities
    predictions_prob = model.predict_proba(x_data)
    #The indexes of each class in the x_data dataset
    idxs_per_class = {key: list(np.where(one_hot_to_int == key)[0]) for key in unique_classes}
    #An empty dictionary where we will fill our results
    results = dict.fromkeys(unique_classes)
    for every_class in idxs_per_class:
        indices = idxs_per_class[every_class]
        counter = 0
        for index in indices:
            y_true = one_hot_to_int[index]
            y_pred = heapq.nlargest(level, range(len(predictions_prob[index])), key=predictions_prob[index].__getitem__)
            if y_true in y_pred:
                counter +=1 
        accuracy = float(counter/len(indices))
        print ('The class {0} with level {1} has accuracy {2}'.format(label[every_class], level, accuracy))
        results[every_class] = accuracy      
        
    return results    
        