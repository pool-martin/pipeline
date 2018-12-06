# 3DCNN experiments on Localization for porn2k dataset

------------------------

Step 0 - Create fold split (outside of container since DL is readonly)
    python create_split_2kporn.py --split-number s1

    checks and create correction for etf limits:
    python check_videos_limits.py

Step 1 - Create container
    save this to ~/.bashrc:
        export OUTSIDE_UID=$(id -u)
        export OUTSIDE_GROUP=$(id -ng)
        export OUTSIDE_GID=$(id -g)

    create local folders and copy data
    - mkdir /work/$USER/DL/2kporn
    - mkdir /work/$USER/Exp/2kporn
    - cp -R ~/DL/2kporn/{videos, etf, folds, eft_frame_count} /work/$USER/DL/2kporn/

    if docker-compose is installed:
        docker-compose up -d --scale gpu=1 gpu
    if docker-compose is not installed:
        sudo bash -c "curl -L https://github.com/docker/compose/releases/download/1.22.0/docker-compose-`uname -s`-`uname -m` > ~/docker-compose"
        chmod +x ~/docker-compose
        ~/docker-compose up -d --scale gpu=1 gpu
    if not using docker-compose:
        docker image build --file ./Dockerfile.gpu --tag jp-pipeline-gpu:v1 .
        nvidia-docker run --name jp_gpu_1 -ti -v /home/jp/Exp/:/Exp/ -v /home/jp/DL/:/DL/ -v .:/workspace/ --userns=host jp-pipeline-gpu:v1 /bin/bash

Step 2 - Create sets (network_training, network_validation, svm_training, svm_validation and test)
    # sample-rate unit = fps
    # sample-length unit = number frames
    # sample-width unit = seconds
    python create_sets.py --split-number s1 --sample-rate 1 --snippet-length 16 --snippet-width 1

Step 3 - Extract features from imagenet/initial weigths
    python train_image_classifier.py --model_name i3d --gpu_to_use 0,1 --num_gpus 2 --batch_size 2 --train=0 --eval=0 --predict=1

Step 3 - Train and extract features
    python train_image_classifier.py --model_name inception_v4 --gpu_to_use 0,1 --batch_size 56 --num_gpus 2 --epochs 26

######################################
SVM
Step 1 - Train svm_train

mkdir /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.models

python train_svm_layer.py --input_training /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/extracted_features/svm_training_set --output_model /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.models/svm_training.model --jobs 4 --svm_method LINEAR_PRIMAL --preprocess NONE --max_iter_hyper 30

Step 2 - Predict

 mkdir /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions

 python predict_svm_layer.py --input_model /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.models/svm_training.model  --input_test /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/extracted_features/svm_validation_set --pool_by_id none  --output_predictions /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/svm_validation.prediction.txt --output_metrics /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/svm_validation.metrics.txt --output_images /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/svm_validation.images --compute_rolling_window --video_split_char _

 python predict_svm_layer.py --input_model /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.models/svm_training.model  --input_test /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/extracted_features/test_set --pool_by_id none  --output_predictions /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/test.prediction.txt --output_metrics /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/test.metrics.txt --output_images /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/test.images --compute_rolling_window --video_split_char _

--------------------------------
Step 4 - Gerando arquivos etf e calculando metricas

mkdir /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf
rm -rf /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf

python results_2_etf.py --output_predictions /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/ml.predictions/test.prediction.txt --output_path /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf --fps_sampled 1 --is_3d --set_to_process test --column k_prob_g5

cd ../trackeval-2014/

perl ./trackeval -error=evt,sum,src -det=det_filename.txt /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf/test/ground_truth/all.txt /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf/test/all.txt >   out-test-gt-2.txt



Step 3 - Analize etf
root@b89952cdbf94:/sva-sw/sms_p7_rd_win/build# python etf_analyze.py /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf/test/ground_truth/all.txt /Exp/2kporn/experiments/i3d/finetune_rmsprop_rgb_imagenet/etf/test/all.txt

