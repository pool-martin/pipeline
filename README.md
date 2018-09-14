# 3DCNN experiments on Localization for porn2k dataset

------------------------

Step 0 - Create fold split (outside of container since DL is readonly)
    python create_split_2kporn.py --split-number s1

Step 1 - Create container
    Export user setting:
        export OUTSIDE_UID=$(id -u)
        export OUTSIDE_GROUP=$(id -ng)
        export OUTSIDE_GID=$(id -g)

    if it's installed:
        docker-compose up -d --scale gpu=1 gpu
    if it's not installed:
        sudo bash -c "curl -L https://github.com/docker/compose/releases/download/1.22.0/docker-compose-`uname -s`-`uname -m` > ~/docker-compose"
        chmod +x ~/docker-compose
        ~/docker-compose up -d --scale gpu=1 gpu
    if not using docker compose:
        docker image build --file ./Dockerfile.gpu --tag jp-pipeline-gpu:v1 .
        nvidia-docker run --name jp_gpu_1 -ti -v /home/jp/Exp/:/Exp/ -v /home/jp/DL/:/DL/ -v .:/workspace/ --userns=host jp-pipeline-gpu:v1 /bin/bash

Step 2 - Create sets (network_training, network_validation, svm_training, svm_validation and test)
    # sample-rate unit = fps
    # sample-length unit = number frames
    # sample-width unit = seconds
    python create_sets.py --split-number s1 --sample-rate 5 --snippet-length 32 --snippet-width 5

Passo 5 - Treinar
    export CUDA_VISIBLE_DEVICES=1
    python train_image_classifier.py --model_name VGG16


			Definir batch, epochSize
			root@b29e6baffa24:/data/torch/ltc# find /data/torch/ltc/datasets/2kporn/rgb/jpg/ -type f | wc -l => 13771673
			epochSize = (13771673/16)/100 => 8608 | está com 4400
			th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_16f_d5_center_crop_new_split -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 4400 -batchSize 100 -cropbeforeresize >> log/process/2kporn_rgb_16f_d5_center_crop_new_split 2>&1
			
			Experimentos time_window fixo:
			th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_16f_time_window_5_d5_center_crop -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 1700 -batchSize 20 -cropbeforeresize -time_window 5 >> log/process/2kporn_rgb_16f_time_window_5_d5_center_crop 2>&1
			
			
			
			th  main.lua -expName teste -nFrames 16 -evaluate -modelNo 20 -cropbeforeresize -nDonkeys 0
			
			
			th  main.lua -nFrames 16 -stream rgb -expName generate_images -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 4400 -batchSize 10 -cropbeforeresize >> log/process/generate_images 2>&1
			
			
jp05-torch-3		th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_16f_d5_tw5_batch50_center_crop -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 680 -batchSize 50 -cropbeforeresize -time_window 5 >> log/process/2kporn_rgb_16f_d5_tw5_batch50_center_crop.txt 2>&1
			
jp05-torch-2		th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_16f_d5_tw1_batch50_center_crop -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 3400 -batchSize 50 -cropbeforeresize -time_window 1 >> log/process/2kporn_rgb_16f_d5_tw1_batch50_center_crop.txt 2>&1

jp05-torch-1		th  main.lua -nFrames 80 -stream rgb -expName 2kporn_rgb_80f_d5_tw5_batch30_center_crop -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 1130 -batchSize 30 -cropbeforeresize -time_window 1 >> log/process/2kporn_rgb_80f_d5_tw5_batch30_center_crop.txt 2>&1

jp05-torch-1		th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_bs25_tw1_ts1_nf16_s16_2 -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 1130 -batchSize 25 -cropbeforeresize -time_window 1 -slide 16 -time_slide 1 >> log/process/2kporn_rgb_bs25_tw1_ts1_nf16_s16_2.txt 2>&1



teste:

python /Exp/scripts/torch/generate_sliding_test_clips.py -w 5 -ts 1 -slide 16 -n 80


	th main.lua -nFrames 80 -stream rgb -expName 2kporn_rgb_80f_d5_center_crop -dataset 2kporn  -dropout 0.5 -batchSize 80 -cropbeforeresize -evaluate -modelNo 26 -slide 16 -time_window 1 -framestep 1
    
    th main.lua -nFrames 80 -stream rgb -expName 2kporn_rgb_80f_d5_center_crop -dataset 2kporn  -dropout 0.5 -batchSize 10 -cropbeforeresize -evaluate -modelNo 1 -slide 16 -time_window 5 -time_slide 1 -framestep 1
    
    
    th  main.lua -nFrames 80 -stream rgb -expName 2kporn_rgb_80f_teste -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 680 -batchSize 1 -cropbeforeresize -time_window 5
    
    
    th  main.lua -nFrames 80 -stream rgb -expName 2kporn_rgb_bs25_tw5_ts1_nf80_s16 -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 1360 -batchSize 25 -cropbeforeresize -time_window 5 -time_slide 1 -slide 16 >> log/process/2kporn_rgb_bs5_ts1_nf80_s162kporn_rgb_bs25_tw5_ts1_nf80_s16.txt 2>&1
    
    th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_bs25_tw1_ts1_nf16_s16 -dataset 2kporn  -dropout 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 2120 -batchSize 80 -cropbeforeresize -time_window 1 -time_slide 1 -slide 16 >> log/process/2kporn_rgb_bs25_tw1_ts1_nf16_s16.txt 2>&1
    
######################################
SVM

torch2pickle to use svm:

python /Exp/torch/ltc/scripts/torchFeats2pythonPickle.py -i /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm_train_1_1_16_16_1_1_16_16_21/feats.txt -o /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.features/svm_train.feats

python /Exp/torch/ltc/scripts/torchFeats2pythonPickle.py -i /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm_validation_1_1_16_16_1_1_16_16_21/feats.txt -o /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.features/svm_validation.feats

python /Exp/torch/ltc/scripts/torchFeats2pythonPickle.py -i /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/test_1_1_16_16_1_1_16_16_21/feats.txt -o /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.features/test.feats


Train svm_train
python train_svm_layer.py --input_training /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.features/svm_train.feats --output_model /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.models/network_train.svm --jobs 4 --svm_method LINEAR_PRIMAL --preprocess NONE --max_iter_hyper 30

Predict

 python predict_svm_layer.py --input_model /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.models/network_train.svm  --input_test /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.features/svm_validation.feats --pool_by_id none  --output_predictions /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/svm_validation.prediction.txt --output_metrics /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/svm_validation.metrics.txt --output_images /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/svm_validation.images --compute_rolling_window

 python predict_svm_layer.py --input_model /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.models/network_train.svm  --input_test /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.features/test.feats --pool_by_id none  --output_predictions /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/test.prediction.txt --output_metrics /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/test.metrics.txt --output_images /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/test.images --compute_rolling_window
 
 
---------------------------------------------------
re-treinando:

jp@dl-05:~$  nvidia-docker attach jp05-torch-4
root@f3074aab93b5:/Exp/torch/ltc# export CUDA_VISIBLE_DEVICES=0
root@f3074aab93b5:/Exp/torch/ltc# th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_bs25_tw1_ts1_nf16_s16_2 -dataset 2kporn  -dropout 0.5 -L                 Rfile LR/UCF101/flow_d5.lua -epochSize 2120 -batchSize 80 -cropbeforeresize -time_window 1 -time_slide 1 -slide 16 >> log/process/2kporn_rgb_bs25_t                 w1_ts1_nf16_s16_2.txt 2>&1


jp@dl-05:~$ nvidia-docker attach jp05-torch-2
root@0b92d7c99c28:/Exp/torch/ltc#
root@0b92d7c99c28:/Exp/torch/ltc# export CUDA_VISIBLE_DEVICES=0
root@0b92d7c99c28:/Exp/torch/ltc# th  main.lua -nFrames 16 -stream rgb -expName 2kporn_rgb_bs25_tw1_ts1_nf16_s16_3_rmsprop -dataset 2kporn  -dropou                 t 0.5 -LRfile LR/UCF101/flow_d5.lua -epochSize 2120 -batchSize 80 -cropbeforeresize -time_window 1 -time_slide 1 -slide 16 -optimMethod rmsprop >>                  log/process/2kporn_rgb_bs25_tw1_ts1_nf16_s16_3_rmsprop.txt 2>&1

root@b89952cdbf94:/sva-sw/sms_p7_rd_win/build# python etf_analyze.py /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/etf/test/ground_truth/all.txt /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/etf/test/all.txt

--------------------------------
Gerando arquivos etf e calculando metricas

rm -rf /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/etf

python results_2_etf.py --output_predictions /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/test.prediction.txt --output_path /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/etf --fps_sampled 1 --is_3d --set_to_process test

cd ../trackeval-2014/

perl ./trackeval -error=evt,sum,src -det=det_filename.txt /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/etf/test/ground_truth/all.txt /Exp/torch/ltc/log/2kporn/2kporn_rgb_bs25_tw1_ts1_nf16_s16/svm.predictions/etf/test/all.txt >   out-test-gt-2.txt

