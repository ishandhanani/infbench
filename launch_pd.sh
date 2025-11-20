cd slurm_runner
python3 submit_job_script.py \
 --model-dir /mnt/lustre01/models/deepseek-r1-0528-fp4-v2 \
 --container-image /mnt/lustre01/users/sgl-baizhou/lmsysorg+sglang+v0.5.5.post2-cu129-arm64.sqsh  \
 --gpus-per-node 4 \
 --config-dir /mnt/lustre01/users/sgl-baizhou/srt-slurm/configs \
 --gpu-type gb200-fp4 \
 --script-variant max-tpt \
 --network-interface enP6p9s0np0 \
 --prefill-nodes 1 \
 --decode-nodes 12 \
 --prefill-workers 1 \
 --decode-workers 1 \
 --account sglang \
 --partition batch \
 --time-limit 4:00:00 \
 --enable-multiple-frontends \
 --num-additional-frontends 9 \
 --benchmark "type=sa-bench; isl=1024; osl=1024; concurrencies=1024x2048x4096; req-rate=inf" \
 --log-dir /mnt/lustre01/users-public/sgl-baizhou/joblogs/max-tpt