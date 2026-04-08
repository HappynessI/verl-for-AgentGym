# TextCraft Main Prefix Datasets

## Layout
- `complete_split/`
  - full cutpoint coverage version
  - keeps all `1496` sampled trajectories
  - prefix sidecars are rebuilt in canonicalized prompt-space
  - does not inherit replay filtering
- `replay_validated/`
  - replay-filtered version
  - fixed-ratio prefixes inherit `new_prefix_rl` replay validation / refine semantics
  - entropy prefixes inherit `entropy_based_prefix` audited validated semantics
  - all datasets still keep `1496` `raw` rows

## Recommendation
- Use `complete_split/` when you want full cutpoint coverage.
- Use `replay_validated/` when you want datasets aligned with the legacy replay-validated full flow.
