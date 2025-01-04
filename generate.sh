python ./generate.py \
    --config_path /scratch/hielab/gbrixi/evo2/nvidia/7b_stripedhyena2_base_4M_resume/shc-evo2-7b-8k-2T-v2.yml \
    --checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/7b_stripedhyena2_base_4M_resume/iter_500000.pt \
    --input_file ebola.txt \
    --temperature 0.5 \
    --cached_generation
# python ./generate.py --config_path /scratch/hielab/gbrixi/evo2/nvidia/100m/shc-evo2-100m-8k-1T.yml --checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/100m/iter_500000.pt
# python ./generate.py --config_path /scratch/hielab/gbrixi/evo2/nvidia/50m/shc-evo2-50m-8k-1T.yml --checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/50m/iter_450000.pt
