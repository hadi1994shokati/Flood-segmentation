
# Description
This repository provides a foundational implementation for flood segmentation in high-resolution aerial imagery.  We present two machine learning models for the segmentation of flood-affected areas using transfer learning techniques: 

* A fine-tuned Segment Anything Model (SAM) comparing the performance of the points prompt versus the bounding box (Bbox) prompt.
   
* A U-Net model employing ResNet-50 and ResNet-101 pre-trained networks as backbones.
   
We use the Flood Area Dataset comprising 290 images with their corresponding masks acquired from [Karim et al., (2022)](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).

The implementation is developed in **Python** using the **PyTorch** framework.

# Requirements
To run the code, ensure you have the following dependencies installed:

* Python 3.8+
* PyTorch 2.0+
* NumPy
* OpenCV
* Additional libraries as listed in [requirements.txt](https://github.com/hadi1994shokati/Flood-segmentation/blob/main/requirements.txt)


# How to Cite
If you use this code in your research, please cite our paper:

Hadi Shokati, Andreas Engelhardt, Kay Seufferheld, Ruhollah Taghizadeh-Mehrjardi, Peter Fiener, Hendrik P.A. Lensch, Thomas Scholten,
Erosion-SAM: Semantic segmentation of soil erosion by water,
CATENA,
Volume 254,
2025,
108954,
ISSN 0341-8162,
https://doi.org/10.1016/j.catena.2025.108954.
