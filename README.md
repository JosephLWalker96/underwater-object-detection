# QR-Code-DL

The datasets folder needs to be put in the underwater-object-detection directory before running.
<br />
The project directory should look like:<br />
underwater-object-detection/
<br />&nbsp;&nbsp;Datasets/
<br />&nbsp;&nbsp;&nbsp;&nbsp;Image/
<br />&nbsp;&nbsp;RCNN
<br />&nbsp;&nbsp;YOLO_script
<br />&nbsp;&nbsp;color-correction-tool
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
mv qr_code.yaml ../yolov5/data/qr_code.yaml<br />
run ./run_yolo.sh<br />
<br />
Remember to change the path in qr_code.yaml accordingly. i.e. path should always be path/to/Datasets/YOLO
<br />
