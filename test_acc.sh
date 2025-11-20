cd slurm_runner 

python3 submit_job_script.py \
 --model-dir /mnt/lustre01/models/deepseek-r1-0528-fp4-v2 \
 --container-image /mnt/lustre01/users/sgl-baizhou/lmsysorg+sglang+v0.5.5.post2-cu129-arm64.sqsh  \
 --gpus-per-node 4 \
 --config-dir /mnt/lustre01/users/sgl-baizhou/srt-slurm/configs \
 --gpu-type gb200-fp4 \
 --script-variant max-tpt-long-isl \
 --prefill-nodes 1 \
 --decode-nodes 12 \
 --prefill-workers 1 \
 --decode-workers 1 \
 --enable-multiple-frontends \
 --num-additional-frontends 9 \
 --benchmark "type=gpqa; num-examples=198; num-threads=512; max-tokens=40000; thinking-mode=deepseek-r1; repeat=10" \
 --log-dir /mnt/lustre01/users-public/sgl-baizhou/joblogs/accuracy

# --benchmark "type=mmlu; num-examples=200" \