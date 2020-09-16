# Nuclei-Segmentation-using-SimpleCNN
A project that was created to segment the nuclei of a microscopic image using Semantic Segmentation concept

Problem definition: 
Segment nuclei from the given microscopic image.

Approach tried:
Training Simple CNN for 'Semantic segmentation':-	

Semantic Segmentation is a method of pixel to pixel mapping where each pixel of an image is assigned a class label.
To know more: https://towardsdatascience.com/semantic-segmentation-with-deep-learning-a-guide-and-code-e52fc8958823

In this approach, we train a simple CNN architecture to segment nuclei from the given microscopic images.

Steps to run the project:

1) Clone this repository
2) Install the packages specified in requirements.txt
3) In test.py change the absolute path of the image to be segmented
4) run test.py using command:

python test.py

Input: 

![val_image](https://user-images.githubusercontent.com/35297458/93341320-95578d00-f84b-11ea-8458-b1c4396605f5.png)

Output:

![val_image_predmask](https://user-images.githubusercontent.com/35297458/93341564-e5365400-f84b-11ea-9d57-64b318aca816.png)
