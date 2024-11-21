Profiling once inside the container:

# Helpful start
```bash
cp /mnt/nv/bin/nsight-systems/target-linux-x64/nsys /opt/app-root/src/bin
```

# Full profile 

```bash
export run_name=full_profile
/mnt/nv/bin/nsight-systems/target-linux-x64/nsys profile --output=/mnt/nsys/output/full_profile --trace=cuda,osrt,cublas,cudnn python /opt/app-root/src/profile.py

oc cp triton-profiling/triton-profiling-job-8wxpz:/mnt/nsys/output/full_profile.nsys-rep full_profile.nsys-rep
```
