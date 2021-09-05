#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time=24:00:00
#SBATCH --mem 200G
#SBATCH --gres=gpu:1,gpumem:12G
#SBATCH --cpus-per-task=12

wandb offline

for i in {1..10}
do
  for embeds_type in normal, average:
  do
    python train_gan.py --model models/ --batch_size 256 --num_epochs 350 --patience 1000 --require_text --embeds_type $embeds_type --exp_name "${embeds_type}Embed_modv2_arm_wh2finger${i}" --learning_rate 0.001 --epochs_train_disc 3 --model v2 --pipeline arm_wh2finger${i}
    for infer_set in "train", "test":
    do
      python inference.py --checkpoint "models/lastCheckpoint_${embeds_type}Embed_modv2_arm_wh2finger${i}.pth" --seqs_to_viz 25 --num_samples 1000 --require_text --embeds_type $embeds_type --infer_set $infer_set --exp_name "${embeds_type}Embed_modv2_arm_wh2finger${i}" --model v2 --pipeline arm_wh2finger${i}
    done
  done
done
