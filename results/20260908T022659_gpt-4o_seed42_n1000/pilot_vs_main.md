# Pilot (20260907T214735_gpt-4o_seed42_n200) vs Main (20260908T022659_gpt-4o_seed42_n1000)

Purpose: check whether the pilot's qualitative pattern replicates at scale -- this is NOT a hypothesis test between sample sizes.

## Metric comparison (overall, pooled across T/N/M)

| Arm | Metric | 200-case | 1000-case | Diff |
| --- | --- | ---: | ---: | ---: |
| A_zero_shot | accuracy | 0.6750 | 0.6890 | +0.0140 |
| A_zero_shot | coverage | 0.9100 | 0.9350 | +0.0250 |
| A_zero_shot | abstention_rate | 0.0900 | 0.0650 | -0.0250 |
| A_zero_shot | span_unsupported_rate_over_evaluable | 0.2600 | 0.2777 | +0.0177 |
| A_zero_shot | semantic_unsupported_rate_over_evaluable | 0.5883 | 0.5943 | +0.0060 |
| A_zero_shot | semantic_supported_accuracy_over_evaluable | 0.2583 | 0.2663 | +0.0080 |
| C_constrained | accuracy | 0.6967 | 0.7020 | +0.0053 |
| C_constrained | coverage | 0.8783 | 0.8893 | +0.0110 |
| C_constrained | abstention_rate | 0.1217 | 0.1107 | -0.0110 |
| C_constrained | span_unsupported_rate_over_evaluable | 0.2067 | 0.1900 | -0.0167 |
| C_constrained | semantic_unsupported_rate_over_evaluable | 0.5267 | 0.5167 | -0.0100 |
| C_constrained | semantic_supported_accuracy_over_evaluable | 0.2833 | 0.2977 | +0.0143 |
| C_plus_unknown | accuracy | 0.6367 | 0.6293 | -0.0073 |
| C_plus_unknown | coverage | 0.7617 | 0.7720 | +0.0103 |
| C_plus_unknown | abstention_rate | 0.2383 | 0.2280 | -0.0103 |
| C_plus_unknown | span_unsupported_rate_over_evaluable | 0.1533 | 0.1650 | +0.0117 |
| C_plus_unknown | semantic_unsupported_rate_over_evaluable | 0.4317 | 0.4397 | +0.0080 |
| C_plus_unknown | semantic_supported_accuracy_over_evaluable | 0.2750 | 0.2750 | +0.0000 |
| D_grounded | accuracy | 0.6283 | 0.6230 | -0.0053 |
| D_grounded | coverage | 0.7983 | 0.7973 | -0.0010 |
| D_grounded | abstention_rate | 0.2017 | 0.2027 | +0.0010 |
| D_grounded | span_unsupported_rate_over_evaluable | 0.1050 | 0.0993 | -0.0057 |
| D_grounded | semantic_unsupported_rate_over_evaluable | 0.4217 | 0.4043 | -0.0173 |
| D_grounded | semantic_supported_accuracy_over_evaluable | 0.2950 | 0.3113 | +0.0163 |

## Replication of pilot findings

**C_constrained has the highest (or near-highest) raw accuracy**

- pilot_200: HOLDS ({'top_arm': 'C_constrained', 'top_accuracy': 0.6966666666666667, 'C_constrained_accuracy': 0.6966666666666667})
- main_1000: HOLDS ({'top_arm': 'C_constrained', 'top_accuracy': 0.702, 'C_constrained_accuracy': 0.702})

**C_constrained's grounding is worse than BOTH C_plus_unknown's and D_grounded's**

- pilot_200: HOLDS ({'C_constrained_semantic_unsupported_rate': 0.5266666666666666, 'comparison_arm_rates': {'C_plus_unknown': 0.43166666666666664, 'D_grounded': 0.4216666666666667}})
- main_1000: HOLDS ({'C_constrained_semantic_unsupported_rate': 0.5166666666666667, 'comparison_arm_rates': {'C_plus_unknown': 0.43966666666666665, 'D_grounded': 0.4043333333333333}})

**C_plus_unknown has a lower semantic-unsupported rate than C_constrained**

- pilot_200: HOLDS ({'C_constrained': 0.5266666666666666, 'C_plus_unknown': 0.43166666666666664})
- main_1000: HOLDS ({'C_constrained': 0.5166666666666667, 'C_plus_unknown': 0.43966666666666665})

**D_grounded has the strongest grounded reliability (highest semantic supported accuracy)**

- pilot_200: HOLDS ({'best_arm': 'D_grounded'})
- main_1000: HOLDS ({'best_arm': 'D_grounded'})

**M-stage raw accuracy is much higher than semantic-supported accuracy, for every arm**

- pilot_200: HOLDS ({'gaps_by_arm': {'A_zero_shot': 0.415, 'C_constrained': 0.45, 'C_plus_unknown': 0.25, 'D_grounded': 0.24}})
- main_1000: HOLDS ({'gaps_by_arm': {'A_zero_shot': 0.435, 'C_constrained': 0.45399999999999996, 'C_plus_unknown': 0.259, 'D_grounded': 0.225}})
