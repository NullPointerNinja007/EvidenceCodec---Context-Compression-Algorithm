# Scripts

Keep shell helpers small and make them invoke Modal wrappers instead of
processing datasets locally. Any command that downloads datasets, writes
checkpoints, or produces benchmark outputs should write to the Modal Volume
mounts configured in `modal_apps/common.py`.
