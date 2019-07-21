# Action recognition by 2D skeleton analysis

### Abstract

This is an implementation of the  techniques  presented  in ["Co-occurrence Feature Learning from Skeleton Data for Action Recognition"](https://arxiv.org/abs/1804.06055) to recognize two-dimensional skeleton using newer technologies.
We worked on the [KTHDataset](http://www.nada.kth.se/cvap/actions/) a  collection  of videos about people performing
six different actions in different scenarios, extracting skeletons thanks to [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) module.

Link to our paper [here](https://drive.google.com/file/d/1-01CTL-k6WWqx98tIsMKwPpJXTKMv4dG/view?usp=sharing).

### Dependencies

- Python 3.x
- Tensorflow 1.9.0
- Keras 2.2.4
- CUDA 9.0
- cuDNN 7.6.0

### Try it out!

We provide some example video to try our network out. To do it, just lunch from console:

`python prediction.py`

at most changing to another of six classes (folder) available inside the file (path variable, line 9).

If you want to use the network for your own data, you can:

- Record a video and:
    1. process it with OpenPose to extract json files. (the commands we
    used are written inside openpose-command file)
    2. write the path into the prediction.py file and launch it
    
- Recognize different classes, training on your own data as explained at the next point "Train"

Examples of boxing and handclapping recorded scenes are shown here:

![](example/example1.gif)

![](example/example2.gif)

### Train

To train the network, firstly use OpenPose to extract jsons from your data, then organize them into folders as follow:

scena/person - classes - sequences - jsons file

then modify these lines:

- train.py
    - line 52 with your dataset path
    - line 53 with your weights path (where to save them)
    - line 54 with your number of different classes
    - if you want to try also cosine normalization
        - line 68 change input tensor dimension
        - line 97 / 128 change the normalization function (from utils.py file)
        - line 98 / 129 change reshape dimensions (selectin from variables at lines 49 / 50)

- utils.py
    - change cross sets name from line 30 with you 'scenas' folder name

and finally launch train.py
    
