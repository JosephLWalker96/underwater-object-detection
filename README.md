# QR-Code-DL

The datasets folder needs to be put in the root directory before running.
<br />
<br />
The project directory should look like:<br />
underwater-object-detection
<br />&nbsp;&nbsp;Datasets/
<br />&nbsp;&nbsp;&nbsp;&nbsp;Image/
<br />&nbsp;&nbsp;RCNN
<br />&nbsp;&nbsp;YOLO_script

<br />
<br />
The Image Folder should look like:<br />
Folder/
<br />&nbsp;&nbsp; Location1/
<br />&nbsp;&nbsp;&nbsp;&nbsp;Camera1/
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image1.png
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image2.jpg
<br />&nbsp;&nbsp;Location2/
<br />&nbsp;&nbsp;&nbsp;&nbsp;Camera2/
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image3.png
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image4.jpg 
<br />
<br />

Faster-RCNN
The project is built on top of the tutorial
in https://www.kaggle.com/havinath/object-detection-using-pytorch-training#Creating-Training-and-Validation-datasets
<br />Running the following command line to run<br />
cd RCNN<br />
./run.sh<br />
<br />

YOLOv5<br />
git clone https://github.com/ultralytics/yolov5.git<br />
cd YOLO_script<br />
run ./run_yolo.sh
