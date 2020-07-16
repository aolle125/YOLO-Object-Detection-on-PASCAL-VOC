# YOLO-Object-Detection-on-PASCAL-VOC

Implementation of a YOLO-like object detector on the PASCAL VOC 2007 dataset.

yolo_loss.py contains the loss function of YOLO.
YOLO predicts multiple bounding boxes per grid cell. 
To compute the loss for the true positive, we only want one of them to be responsible for the object.
For this purpose, we select the one with the highest IoU (intersection over union) with the ground truth. 
This strategy leads to specialization among the bounding box predictions. Each prediction gets better at predicting certain sizes and aspect ratios.
YOLO uses sum-squared error between the predictions and the ground truth to calculate loss. 

The loss function composes of:
the classification loss.
the localization loss (errors between the predicted boundary box and the ground truth).
the confidence loss (the objectness of the box).
 
 
 resnet_yolo.py contains a pre-trained network structure for the model. The network structure has been inspired by DetNet.
 
 MP2_P2.ipynb contains the training a results of our object detection.
