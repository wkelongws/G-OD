
I have done some experiments on my machine (GPU: 1 x TITAN X; CPU: 12 x Intel(R) Core(TM) i7-3930K CPU @ 3.20GHz).

I tested the reference speed of all models in tensorflow detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) on the two provided sample testing images: image1 is "two dogs" and image2 is "kiting on beach". In addition, to figure out whether the timeline recording is slowing down the reference speed or not, I did the experiments both with and without timeline recording.

Here is a brief view of my experiments:




Some of my findings based on there experiments:

1. Reference speed is highly depend on testing images, especially in GPU case. (I am not sure why.)
2. GPU does accelerate the reference speed a lot comparing to CPU. But the speed is still most of the time lower than the reference speed reported by Google (also on TITAN X).
3. On my machine faster_rcnn_nas_coco doesn't perform significantly slower than other models. This observation is inconsistent with Google's report.

The detailed experiment results can be found in this google drive folder: https://drive.google.com/open?id=1VimHDybeeLan2Zn_C9n3LDPVVm5HTOG-
The folder contains:
1. A spread sheet result summary
2. Two saved plots
3. Two test images
4. Two folders, CPU and GPU, contain the chrome trace json files for all the individual experiments. 52 Json files in total.

Hope these information can help.

Best.

Shuo

how to profile tensorflow: https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d




python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /home/shuo/Shuo/tensorflow_models/research/object_detection/faster_rcnn_resnet101_coco_2017_11_08/pipeline.config \
    --trained_checkpoint_prefix /home/shuo/Shuo/tensorflow_models/research/object_detection/faster_rcnn_resnet101_coco_2017_11_08/model.ckpt \
    --output_directory output_inference_graph.pb
