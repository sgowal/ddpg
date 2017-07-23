# Privacy

This folder contains code related to the work on private trajectories.
Feel free to ignore (unless you want to reproduce the results shown in "TODO").

## Command

```bash
python run.py \
  --output_directory ../ddpg_experiments/test --force \
  --environment=Private-Cart-v0 \
  --options="evaluate_after_timesteps: 2000 max_timesteps: 20000 device: '/gpu:0'"
```
