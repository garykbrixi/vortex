# echo evo2_50m_gen
# python test/accuracy_test_forward.py \
#     --config_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/50m/shc-evo2-50m-8k-1T.yml \
#     --checkpoint_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/50m/iter_450000.pt

# echo evo2_100m_gen
# python test/accuracy_test_forward.py \
#     --config_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/100m/shc-evo2-100m-8k-1T.yml \
#     --checkpoint_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/100m/iter_500000.pt

# echo evo2_1b_gen 
# python test/accuracy_test_forward.py \
#     --config_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/1b/shc-evo2-1b-8k-1T.yml \
#     --checkpoint_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/1b/iter_490000.pt


echo evo2_50m_gen
python test/accuracy_test_generation.py \
    --config_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/50m/shc-evo2-50m-8k-1T.yml \
    --checkpoint_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/50m/iter_450000.pt \
    --top_k 1

echo evo2_100m_gen
python test/accuracy_test_generation.py \
    --config_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/100m/shc-evo2-100m-8k-1T.yml \
    --checkpoint_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/100m/iter_500000.pt \
    --top_k 1

echo evo2_1b_gen 
python test/accuracy_test_generation.py \
    --config_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/1b/shc-evo2-1b-8k-1T.yml \
    --checkpoint_path /scratch/hielab/gbrixi/evo2/vortex_interleaved/1b/iter_490000.pt \
    --top_k 1

# generation test accuracy with greedy decoding on evo2_100m is 64.25, for 1B is 85.5