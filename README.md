# realtime_face_recognition


this project depend on python 2.7,opencv,facenet project,and the facenet use 128 dimension vector model.


Reference project:


1,facenet github address https://github.com/davidsandberg/facenet


2,Video-stream-face-recognition https://github.com/junjun870325/Video-stream-face-recognition

How to use the project


1,Download the pretrained facenet model,put it in the facenet_model directory

download address https://pan.baidu.com/s/1cnd1rvD9-FOQCbX4Lx57-A


2,Put the feature data in model_train_data directory,the formate as below:

feature_data,each person got one dirctory only,every image is 160 X 160 pixel

how to get 160x160 image refer to https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1

feature_data dirctory structure

--->Abby

   Abby1.jpg

   Abby2.jpg
    
--->John

   John1.jpg
    
   John2.jpg

3,Run the classifer_model.py,it will use the pretrain facenet model and feature data to build the classifer,after fininshed, save the classifer to modle/feature_models.pki file,at last,test the accuracy.


4,open the realtime_recognite.py file,set the input_path varible's vale to your test video file（if the value is 0,mean use the camera)


5,run realtime_recognitie.py,then you can see the recognition process.


本项目依赖于python 2.7,opencv,facenet,并且facenet 使用128维的特征向量


参考项目：


1,facenet github address:https://github.com/davidsandberg/facenet


2,https://github.com/junjun870325/Video-stream-face-recognition


怎样运行项目    


1，下载预训练的facent模型，下载完后放置在facenet_model文件夹下


2，将特征数据放到model_train_data目录下，特征数据的格式如下:

特征数据中每个人的数据在一个单独的文件夹内，每个图像文件都是 160 X 160像素

怎样获取160 X 160像素的图像文件可以参考 https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1

特征数据文件夹结构示意如下

--->Abby

   Abby1.jpg

   Abby2.jpg
    
--->John

   John1.jpg
    
   John2.jpg
       
       
3，运行classifer_model.py，该文件将使用预训练模型和特征数据用于构建分类器，构建完毕后，分类模型保存在model/feature_models.pki文件中，并在最后测试了模型的精度。


4，打开realtime_recognite.py文件，设置input_path 变量的值为用于测试的文件名（如果是数字0，那代表使用默认摄像头)


5，运行realtime_recognitie.py,你将看到识别过程。








