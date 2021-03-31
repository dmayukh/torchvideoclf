# torchvideoclf

### Video classification using pytorch

The code in this repo is largely a reuse of the pytorch vision video classification code from here https://github.com/pytorch/vision.git

While `vision/references/video_classification/train.py` in the pytorch repo uses PyAV to process the videos, here we do not use PyAV, we instead use a sequence of image files to create the training dataset.
The downloader downloads videos from youtube as a collection of images and also prepares an annotation file.

The train.py uses the image collections to prepare the training dataset.

You code in this repo was developed on this docker image *mayukhd/torch1.7:videoclassification*

`docker pull mayukhd/torch1.7:videoclassification`

You should run the below in the above container

#### Steps:

- Prepare the source video list, the ones we wish to download from YouTube and tag them appropriately
    - Each entry in the video list needs to be of the format: 
    
    > `{'url':"\<url of the video>", 'category':'\<category>', 'start': \<start seconds>, 'end': \<end seconds>}`
     
     e.g., the list file should look like
     
    > `[{'url':"\<url>", 'category': "\<cat>", 'start': 506, 'end': 508},
        {'url':"\<url>", 'category': "\<cat>", 'start': 123, 'end': 127}]`

- Download the images from YouTube using the downloader utility
  - Run this in the container 
  
  `python3 download.py --train_video_list=<full path to the training list> 
  --dataset_traindir=<full path to where the image sequences for training should be saved> 
  --val_video_list=<full path to the test list> --dataset_valdir=<full path to where the image sequences for validation should be saved>`

- Run the train.py to train the model on the images we downloaded
  - The code used GPU by default, you can change it via the --device parameter when running
    
    `python3 train.py --train-dir=dataset/train --val-dir=dataset/val --output-dir=checkpoint --pretrained`
    
    export port 6006 in the container
    
    run `tensorboard --logdir=runs` in another session 
    
    goto https://\<url>:6006 to view the training metrics
    
    ![image](https://user-images.githubusercontent.com/17194414/113135283-92e4df80-923f-11eb-81cd-b0074b34cb3c.png)

- Run a test
