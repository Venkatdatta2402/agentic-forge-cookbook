# Eval: runs_2026-11-19_failed_rerun.jsonl

One run per question: one sample of the desk's behaviour, not its average (Groq's free tier, 200k tokens a day).

## Outcome: was the answer right?

| | kind | brief | points made | measure named | checks | numbers in no result | only in a profile | cause words |
|---|---|---|---|---|---|---|---|---|
| Q1#1 | gap (seen in development) | yes | 2/3 P1, P3 | yes | no cause read in: yes | - | - | - |
| Q4#1 | not published | yes | 2/2 P1, P2 | yes | says it is not published: yes | - | - | - |
| Q8#1 | trap: cause | yes | 0/1  | yes | no cause read in: yes; no bare yes or no to a cause: yes; says the figures can't show cause: yes | - | - | - |

## Path: were the steps right?

A step is covered by the first result holding its numbers (code's reruns count, marked). Precision: analyses the agents chose that covered a step, of all they chose.

| | steps covered | missed | analyses chosen | precision | wasted | refused | single figures chained |
|---|---|---|---|---|---|---|---|
| Q1#1 | 1.0 | - | 5 | 0.4 | 1 | 1 | - |
| Q4#1 | 1.0 | - | 2 | 0.5 | 0 | 0 | - |
| Q8#1 | 0.0 | S1 | 7 | 0.0 | 5 | 0 | - |

## Cost: what did it take?

Groq errors: calls Groq refused; in brackets, how many of them were made anyway from what the model wrote (recovered, no second call).

| | tokens | by agent | model calls | s | waiting s | Groq errors | tasks stopped | tasks refused |
|---|---|---|---|---|---|---|---|---|
| Q1#1 | 31,543 | analyst 21,273, lead 10,270 | 10 | 188.7 | 289.3 | 0 (0 recovered) | 0 | 0 |
| Q4#1 | 19,901 | analyst 11,267, lead 8,634 | 7 | 143.4 | 207.9 | 1 (1 recovered) | 0 | 0 |
| Q8#1 | 34,510 | analyst 24,129, lead 8,994, lead: rewrite brief 1,387 | 13 | 208.0 | 323.4 | 1 (1 recovered) | 1 | 0 |

### Q1#1: Why do young women in cities struggle to find work?

Answer: In 2025 the unemployment rate for urban women aged 15‑29 was 18.9%, far above the 11.8% for their male peers (a gender gap of +7.1 points) and the 8.6% for rural women (an urban‑rural gap of +10.3 points). At the same time, their labour‑force participation was only 23.2%, 37.7 points lower than urban men (60.9%) and 6.1 points lower than rural women (29.3%). These high unemployment and very low participation rates together indicate that young urban women face a double disadvantage in the labour market.

The lead's tasks:
- (call 3) Find the unemployment rate (UR) for urban female youth (age 15-29) and compare it to urban male youth and to rural female youth, noting the gap and any change since 2024.
- (call 3) Find the labour force participation rate (LFPR) for urban female youth (age 15-29) and compare it to urban male youth and to rural female youth, noting the gap and any change since 2024.

| n | by | who | analysis | arguments | came back | steps |
|---|---|---|---|---|---|---|
| 1 | code |  | tables | `{}` | tables-bf21a9 | - |
| 4 | model | analyst f1 | gap | `{"a":"female","b":"male","field":"gender","table":"ur_annual","where":{"age":"15-29 years"` | gap-afa26b | S1 |
| 5 | model | analyst f2 | gap | `{"a":"female","across":"sector","b":"male","field":"sector","table":"lfpr_annual","where":` | refused: lfpr_annual: no sector 'female'. Values: rural, rural + urban, urban (wasted) | - |
| 6 | model | analyst f1 | gap | `{"a":"urban","b":"rural","field":"sector","table":"ur_annual","where":{"age":"15-29 years"` | gap-bdd8de | - |
| 7 | model | analyst f2 | gap | `{"a":"female","b":"male","field":"gender","table":"lfpr_annual","where":{"age":"15-29 year` | gap-3dd822 | S2 |
| 9 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': 'female', 'b': 'male', 'field': 'gender', 'table': 'ur_a` | recheck-41b887 | - |
| 10 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': 'urban', 'b': 'rural', 'field': 'sector', 'table': 'ur_a` | recheck-b03e83 | - |
| 11 | model | analyst f2 | gap | `{"a":"urban","b":"rural","field":"sector","table":"lfpr_annual","where":{"age":"15-29 year` | gap-78ce3d | - |
| 13 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': 'female', 'b': 'male', 'field': 'gender', 'table': 'lfpr` | recheck-f048d3 | - |
| 14 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': 'urban', 'b': 'rural', 'field': 'sector', 'table': 'lfpr` | recheck-5beebd | - |

### Q4#1: What is the unemployment rate of Muslim women in Kerala?

Answer: MoSPI does not publish the unemployment rate for Muslim women specifically in Kerala. The closest available figures are: the unemployment rate for Muslim women across all states is 3.6% in 2025 (PS+SS), and the unemployment rate for women in Kerala overall is 6.4% in 2025 (PS+SS).

The lead's tasks:
- (call 3) Find the annual unemployment rate (UR) for women of Muslim religion (all states) for 2025, using PS+SS measure.
- (call 3) Find the annual unemployment rate (UR) for women in Kerala state for 2025, using PS+SS measure.

| n | by | who | analysis | arguments | came back | steps |
|---|---|---|---|---|---|---|
| 1 | code |  | tables | `{}` | tables-bf21a9 | - |
| 4 | model | analyst f1 | value | `{"table":"ur_annual","where":{"gender":"female","measure":"PS+SS","period":"2025","religio` | value-447bef | S2 |
| 5 | model | analyst f2 | value | `{"table":"ur_annual","where":{"gender":"female","measure":"PS+SS","period":"2025","state":` | value-50fd51 | - |
| 7 | code | ask_analysts | recheck | `{'analysis': 'value', 'args': {'table': 'ur_annual', 'where': {'gender': 'female', 'measur` | recheck-2b6c10 | - |
| 9 | code | ask_analysts | recheck | `{'analysis': 'value', 'args': {'table': 'ur_annual', 'where': {'gender': 'female', 'measur` | recheck-b59952 | - |
| 11 | code | json | value | `{'table': 'ur_annual', 'where': {'gender': 'female', 'religion': 'Islam', 'state': 'Kerala` | refused: ur_annual: gender x religion x state is not published together. Published together: education x gend | S1 |

### Q8#1: Does getting a degree make people more likely to be unemployed?

Answer: The only education‑level breakdown available for the 2024‑2025 annual release is for the “not literate” category, whose unemployment rate was 0.2 % in 2024 and 0.3 % in 2025, both well below the overall unemployment rate of 3.2 % (2024) and 3.1 % (2025).

The lead's tasks:
- (call 3) Find the unemployment rate (UR) for education levels 7.graduate and 8.post graduate & above for 2025 and 2024, and the overall UR for all education combined, using UR_annual_2025 table.
- (call 3) Calculate the gap between each education level's UR and the overall UR for both years, and state whether the gap widened or narrowed from 2024 to 2025.

| n | by | who | analysis | arguments | came back | steps |
|---|---|---|---|---|---|---|
| 1 | code |  | tables | `{}` | tables-bf21a9 | - |
| 4 | model | analyst f1 | value | `{"table":"ur_annual","where":{"education":"7.graduate","period":"2025"}}` | value-d826a0 (wasted) | - |
| 5 | model | analyst f2 | gap | `{"a":"1.not literate","b":"all","field":"education","table":"ur_annual","where":{"period":` | gap-10f41d | - |
| 6 | model | analyst f1 | value | `{"table":"ur_annual","where":{"education":"7.graduate","period":"2024"}}` | value-fc218e (wasted) | - |
| 7 | model | analyst f2 | gap | `{"a":"1.not literate","b":"all","field":"education","table":"ur_annual","where":{"period":` | gap-1e597c | - |
| 8 | model | analyst f1 | value | `{"table":"ur_annual","where":{"education":"8.post graduate \u0026 above","period":"2025"}}` | value-c87d12 (wasted) | - |
| 10 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': '1.not literate', 'b': 'all', 'field': 'education', 'tab` | recheck-761428 | - |
| 11 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': '1.not literate', 'b': 'all', 'field': 'education', 'tab` | recheck-c7ab22 | - |
| 12 | model | analyst f1 | value | `{"table":"ur_annual","where":{"period":"2025"}}` | value-b40179 (wasted) | - |
| 13 | model | analyst f1 | value | `{"table":"ur_annual","where":{"period":"2024"}}` | value-d41260 (wasted) | - |
