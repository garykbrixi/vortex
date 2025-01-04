
# python ./generate.py --config_path /scratch/hielab/gbrixi/evo2/nvidia/7b_stripedhyena2_base_4M_resume/shc-evo2-7b-8k-2T-v2.yml --checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/7b_stripedhyena2_base_4M_resume/iter_500000.pt
# python ./generate.py --config_path /scratch/hielab/gbrixi/evo2/nvidia/100m/shc-evo2-100m-8k-1T.yml --checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/100m/iter_500000.pt
# python ./generate.py --config_path /scratch/hielab/gbrixi/evo2/nvidia/50m/shc-evo2-50m-8k-1T.yml --checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/50m/iter_450000.pt
# 
python -m pdb ./generate_speculative.py \
    --target_config_path /scratch/hielab/gbrixi/evo2/nvidia/50m/shc-evo2-50m-8k-1T.yml \
    --target_checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/50m/iter_450000.pt \
    --draft_config_path /scratch/hielab/gbrixi/evo2/nvidia/100m/shc-evo2-100m-8k-1T.yml \
    --draft_checkpoint_path /scratch/hielab/gbrixi/evo2/nvidia/100m/iter_500000.pt \
    --gamma 4 \
    --input_file ./prompt.txt \
    --debug
    
    # --cached_generation \

