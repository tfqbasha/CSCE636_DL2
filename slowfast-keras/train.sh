#!/bin/bash

python train.py --root_path=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/slowfast-keras2Class/slowfast-keras \
    --video_path=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/Data/data_clips_100_ref_jpg \
    --name_path=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/Data/data_clips_100_ref/classInd.txt \
    --train_list=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/Data/data_clips_100_ref/train.txt \
    --val_list=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/Data/data_clips_100_ref/test.txt \
    --data_name=ucf_101 \
    --num_classes=2 \
    --workers=4 \
    --batch_size=4 \
    --crop_size=224 \
    --clip_len=32 \
    --short_side 256 320 \
    --test_videos_path=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/Data/data_clips_100_ref_test_jpg \
    --test_list_path=/content/drive/My\ Drive/Colab\ Notebooks/CSCE636/Data/data_clips_100_ref/test_final.txt