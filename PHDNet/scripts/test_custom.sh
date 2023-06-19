#!/usr/bin/env bash
DISPLAY_PORT=8097


G='phd'
loadSize=256


# network design
is_matting=1
patch_size=4
is_skip=1
is_fft=1
fft_num=3

batchs=1
test_epoch=latest

#####network design
model_name=phdnet
datasetmode=phd
content=""  # content dir or name. If is dir, the triplets `content_image/style_image/mask` should be the same name.
style=""  # style dir or name
mask=""  # mask dir or name
save_dir="../samples"
NAME="phdnetadv_Gphd_skip1_fft1-num3_Dconv-patch4_Content2_Style1_Ggan10_lr2e-4_batch4"
checkpoint=''../1207-blendC/''

CMD="python ../test_custom.py \
--name $NAME \
--checkpoints_dir $checkpoint \
--model $model_name \
--netG $G \
--dataset_mode $datasetmode \
--content $content \
--style $style \
--mask $mask \
--save_dir $save_dir \
--is_train 0 \
--display_id 0 \
--gan_mode wgangp \
--normD batch \
--normG batch \
--preprocess none \
--input_nc 3 \
--batch_size $batchs \
--num_threads 6 \
--print_freq 400 \
--display_freq 1 \
--gpu_ids 2 \
--load_size $loadSize \
--is_matting $is_matting \
--is_skip $is_skip \
--is_fft $is_fft \
--epoch $test_epoch  \
--patch_number $patch_size \
--fft_num $fft_num \
"
echo $CMD
eval $CMD
