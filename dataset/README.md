# Dataset

Two datasets for coordinator routing, 1650 samples each:

- `single_turn.csv`: one developer request per sample
- `multi_turn.csv`: a five-messages conversation per sample, routed at the last turn

Each sample is labelled with one of five sub-agents (experts): `time_to_market`,
`scale_arch`, `resource_opt`, `legacy_compat`, `strict_compliance`. The label is the
constraint that is binding in the request: exactly one is a binding, and every other
constraint is a waived. The full resolution rule is `RESOLUTION_RULE` in
`src/dataset/config.py`.

## Columns

| Column | Meaning |
| --- | --- |
| `id` | sample id |
| `request_id` | shared by the two samples of an adversarial pair |
| `kind` | `single_turn` or `multi_turn` |
| `scenario_id` | the seed situation the sample is built from (`src/dataset/scenarios.json`); group splits by it |
| `construction` | `plain`, `natural` or `adversarial` |
| `gold` | the correct sub-agent |
| `decoy` | the competing sub-agent the request also mentions (empty for `plain`) |
| `collision` | `gold>decoy`, or `none` for `plain` |
| `persona` | writing style |
| `pattern` | how the mandate is hidden in the request (`adversarial` only) |
| `variant` | `a` or `b`, the two halves of an adversarial pair |
| `pivot` | the phrase that differs between the two halves of a pair and flips the label |
| `oracle_predicted` | sub-agent chosen by the LLM judge applying the resolution rule |
| `oracle_correct` | `oracle_predicted` equals `gold` |
| `oracle_decidable` | the judge found one binding constraint |
| `oracle_quote` | the text the judge cited as evidence |
| `oracle_constraints` | the constraints the judge found in the text |
| `pair_overlap` | word overlap with the other half of the pair |
| `review_decision` | outcome of the manual review, if the sample was reviewed |
| `regenerate` | `1` if the sample was flagged for regeneration (always `0` here) |
| `n_turns` | number of turns (1 or 5) |
| `turns` | the request as a JSON list of chat messages, the model input |
| `text` | readable transcript, one line per turn |
| `user_text` | developer turns only, concatenated |

## License

[CC BY 4.0](LICENSE)
