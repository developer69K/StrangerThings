# KaggleRuns

Kaggle Competitions and Kernel Runs

## DataScience Bowl

### Sources to Read for Nuclei detection
 + Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow
    + https://github.com/matterport/Mask_RCNN#installation
  + Kaggle DS Bowl Baseline
    + https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline
  + Fast AI
    + https://github.com/fastai/courses/tree/master/setup
  + U-net architecture
    + https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
  + Dog breed classification - https://towardsdatascience.com/dog-breed-classification-hands-on-approach-b5e4f88c333e  
  + YOLO architecture - https://github.com/experiencor/basic-yolo-keras/blob/master/examples/Blood%20Cell%20Detection.ipynb
  + Legacy U-net architecture - The Model is based on https://arxiv.org/abs/1505.04597 , seems the unet architecture came up in 2015 and was considered best at that time for Biomedical image segmentation http://www.cs.cmu.edu/~jeanoh/16-785/papers/ronnenberger-miccai2015-u-net.pdf
  + Using Pytorch for doing this - https://tuatini.me/practical-image-segmentation-with-unet/

## Some resources about loading data into Amazon s3 and using Amazon Sagemaker
  + https://github.com/keithweaver
  + https://github.com/keithweaver/python-aws-s3/
  + Using Amazon Sagemaker: https://github.com/aws/sagemaker-spark/tree/master/sagemaker-pyspark-sdk
  + tqdm : https://pypi.python.org/pypi/tqdm
  + image reading : http://www.scipy-lectures.org/advanced/image_processing/
  + https://docs.aws.amazon.com/sagemaker/latest/dg/apache-spark-example1.html

## Some other resources
  + https://stackoverflow.com/questions/13584118/how-to-write-a-path-with-latex
  + https://stackoverflow.com/questions/9419162/python-download-returned-zip-file-from-url
  + https://cs.stackexchange.com/questions/51387/what-is-the-difference-between-object-detection-semantic-segmentation-and-local
  + http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
  + Gluon/mxnet - http://gluon.mxnet.io/
  + Pytorch - http://pytorch.org/tutorials/
  + Torch - http://torch.ch/
  + BCE - https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/22361
  + AWS: wget https://s3-us-west-2.amazonaws.com/downloaditems-oregon/churn.txt
    >(target * K.log(output) + (1.0 - target) * K.log(1.0 - output))

  + Boto3: Amazon SDK for Python : https://aws.amazon.com/sdk-for-python/
  + Amazon ML : https://aws.amazon.com/documentation/machine-learning/
  + Comparisions : http://codeinpython.com/tutorials/deep-learning-tensorflow-keras-pytorch/
  + Amazon AMIs for Deep learning - https://aws.amazon.com/machine-learning/amis/
  + Fast.ai forum - http://forums.fast.ai/t/implementing-mask-r-cnn/2234
  + Deep Residual networks - https://github.com/KaimingHe/deep-residual-networks
  + Pytorch using DL - http://torch.ch/blog/2016/02/04/resnets.html
  + Scikit-Image(morphology) - http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.label
  +

##  Tensorflow examples
  + https://www.programcreek.com/python/example/90553/tensorflow.one_hot
  + http://tflearn.org/metrics/
  + https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d

## Project guidelines
  + (https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb)
  + Boto3 - http://boto3.readthedocs.io/en/latest/reference/services/s3.html


## Other options for ML and running jupyter notebooks
  + https://www.paperspace.com/
  + https://www.floydhub.com/

## floydhub commands
 ```
 floyd init my_jupyter_project
 floyd run --data sananand007/datasets/datasciencebowl2018/2 --gpu --mode jupyter
 ```  
