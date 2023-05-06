## Cloud Type Classifier

An image classifier and inference deployment for classifying cloud type from the 10 main types.

The cloud_learner uses a collection of images downloaded from the web to train a neural network with the 10 main cloud types:

    Cirrus
    Cirrocumulus
    Cirrostratus
    Altocumulus
    Altostatus
    Nimbostratus
    Stratocumulus
    Stratus
    Cumulus
    Cumulonimbus

The neural network is pre-trained with resnet34. This speeds up the learning proccess.

The model has a error rate of 0.278 at present. An draw back is the limited data used to train the model. Improvments would need a larger training dataset. It has the following confusion matrix:
<img src="images/cloud_confusion_matrix.jpg" style="border-radius:5%">

Example webapp cloud type classifier:
<img src="images/cloud_demo.jpg" style="border-radius:5%">