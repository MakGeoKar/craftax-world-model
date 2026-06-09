# Experiment Summary

Environment: `Craftax-Classic-Pixels-v1`  
Budget: 1M environment steps  
Seeds: default/0, 42, 239  
Metric: final logged score / return

| Method | Score last | Return last | Notes |
|---|---:|---:|---|
| PPO baseline | 1.458953 | 3.063586 | Mean over 3 matched seeds |
| WM-PPO | 1.557813 | 3.276904 | Mean over 3 matched seeds |
| WM-PPO + imagined actor | worse than WM-PPO on seeds 0,42 | worse / similar | Did not justify extra compute |

Main result:

- WM-PPO improved score by `+0.098860` / `+6.78%`.
- WM-PPO improved return by `+0.213318` / `+6.96%`.
- Explicit imagined actor/value objectives were less stable and did not outperform the simpler prediction-error exploration setup.
