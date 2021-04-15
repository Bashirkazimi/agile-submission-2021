# Extraction of linear structures from digital terrain models using deep learning

This repository accompanies our AGILE 2021 paper submission titled "Extraction of linear structures from digital terrain models using deep learning". Deep learning models including HRNet and SegNet are used for segmentation of linear structures in Digital Terrain Models (DTMs). The training and validation dataset is private to Lower Saxony State Office for Heritage. However, the trained models and the test dataset to which the reported results correspond to are included here. Results can be reproduced using the following steps.



*   Clone this repository and navigate to it on your computer.
*   Install required libraries using:
    `pip install -r requirements.txt`

*   Download the test data from this [link](https://seafile.cloud.uni-hannover.de/d/95a74b9a5b0e4e639077/). Extract and copy it to the root of this repo. 

*   Create test examples from the large DTM and its corresponding label that you downloaded: 
    *   `python3 create_dataset.py`
*   Evaluate binary segmentation with HRNet:

    *   `cd HRNetBinarySegmentation`
    *   `python3 evaluate.py --evaluation_file=evaluation_file.csv`

*   Evaluate binary segmentation with SegNet:
    *   `cd SegNetBinarySegmentation`
    *   `python3 evaluate.py --evaluation_file=evaluation_file.csv`

*   Evaluate multiclass segmentation with HRNet:

    *   `cd multiclassSegmentation`
    *   `python3 evaluate.py --evaluation_file=evaluation_file.csv`

For each of the evaluations above, the models are created and the training weights are loaded from the weight files under the `files` folder for each experiment. The models are evaluated on the test data and the results are written to `evaluation_file.csv` saved in the mentioned `files` folder. It contains precision, recall, F1-score, and Intersection Over Union (IOU) scores for each model on the test data.

Training and validation datasets (not included here) are also created similarly to the test data explained above. Training is done using the following command from each experiment folder:

`python3 train_segmenter.py`


The trained models can be used to scan large DTMs in a sliding window fashion making predictions and saving the results to vector files (.shp). It is done using the following command from each experiment folder:



```
python3 predict_sliding.py --output_shp_file=vector.shp --input_dtm=test_dtm.tif
```

It runs the trained model on the `test_dtm.tif` raster file and saves the predictions into `vector.shp` vector file in the corresponding `files` folder.

The above commands are run using the default arguments/parameters. Further details of parameters for training, validation and testing experiments are including in the `config.py` file for each experiment. 

Training history and evaluation results on the test set are already included csv files in the `files` folder for each experiment in case you do not want to or do not have the resources to run the above commands. Vector files for predictions on the large region are also included. 
