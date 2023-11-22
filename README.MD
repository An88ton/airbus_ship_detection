This is the project for airbus ship detection challenge https://www.kaggle.com/competitions/airbus-ship-detection/overview . Main purpose of trained dnn is finding ships on satelites images.
Firstly you need to download dataset, for this you need to have kaggle CLI, as you have it execute this comand
<br/>```kaggle competitions download -c airbus-ship-detection``` <br />
In created folder  you will find folder train_v2 and train_ship_segmentation.cvs, in folder stored images, in csv file coordinates of pixel for masks. 
Also present folder test_v2 and corresponding file sample_submission_v2.csv, with cordinates of pixel, but data in csv broken so all lines for encoded pixels have values 1 2, so it will not be used during training. 
Test set will be created from images in training folder.
<br/>
To create masks run ```py ./create_mask.py```. there are flag ```isKeepEmpty``` by default false, if you set it to true id will delete part of empty images. Because presents of empty images too high and it affect results of the model.
Model trained in file ```main.py``` to start trainig run ```py main.py```