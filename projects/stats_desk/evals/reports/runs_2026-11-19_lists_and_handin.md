# Eval: runs_2026-11-19_lists_and_handin.jsonl

One run per question: one sample of the desk's behaviour, not its average (Groq's free tier, 200k tokens a day).

## Outcome: was the answer right?

| | kind | brief | points made | measure named | checks | numbers in no result | only in a profile | cause words |
|---|---|---|---|---|---|---|---|---|
| Q1#1 | gap (seen in development) | yes | 2/3 P1, P3 | yes | no cause read in: yes | - | - | - |
| Q4#1 | not published | yes | 2/2 P1, P2 | yes | says it is not published: yes | - | - | - |
| Q8#1 | trap: cause | yes | 1/1 P1 | yes | no cause read in: yes; no bare yes or no to a cause: yes; says the figures can't show cause: yes | - | - | - |

## Path: were the steps right?

A step is covered by the first result holding its numbers (code's reruns count, marked). Precision: analyses the agents chose that covered a step, of all they chose.

| | steps covered | missed | analyses chosen | precision | wasted | refused | single figures chained |
|---|---|---|---|---|---|---|---|
| Q1#1 | 1.0 | - | 2 | 1.0 | 0 | 0 | - |
| Q4#1 | 1.0 | - | 2 | 0.5 | 0 | 0 | - |
| Q8#1 | 0.0 | S1 | 5 | 0.0 | 4 | 0 | - |

## Cost: what did it take?

Groq errors: calls Groq refused; in brackets, how many of them were made anyway from what the model wrote (recovered, no second call).

| | tokens | by agent | model calls | s | waiting s | Groq errors | tasks stopped | tasks refused |
|---|---|---|---|---|---|---|---|---|
| Q1#1 | 24,231 | analyst 13,690, lead 10,541 | 7 | 134.3 | 199.7 | 0 (0 recovered) | 0 | 0 |
| Q4#1 | 19,644 | analyst 11,431, lead 8,213 | 7 | 145.7 | 211.7 | 0 (0 recovered) | 0 | 0 |
| Q8#1 | 38,182 | analyst 26,157, lead 12,025 | 12 | 280.9 | 412.7 | 2 (1 recovered) | 1 | 0 |

### Q1#1: Why do young women in cities struggle to find work?

Answer: Young women in urban areas have very low labour market attachment: in 2025 only 23.2% of urban 15‑29‑year‑old women were in the labour force versus 60.9% of men, a gender gap of –37.7 points (narrowed slightly by 0.3 points since 2024). At the same time their unemployment is higher, with an urban female‑male gap of +7.1 percentage points (women 18.9% vs men 11.8%), which widened by 0.3 points since 2024. The largest gaps are seen in states such as Dadra & Nagar Haveli & Daman & Diu (LFPR gap –53.1 points) and Bihar (UR gap +27.1 points). These figures show that young urban women are far less likely to be participating in work and, when they do, face higher unemployment, indicating a struggle to find work.

The lead's tasks:
- (call 3) Find the LFPR for age 15-29, sector urban, by gender and state for 2025 and 2024; compute the gender gap (women minus men) overall and list states with the largest negative gaps; note change in the gap since 2024.
- (call 3) Find the UR for age 15-29, sector urban, by gender and state for 2025 and 2024; compute the gender gap (women minus men) overall and list states with the largest positive gaps (higher female unemployment); note change in the gap since 2024.

| n | by | who | analysis | arguments | came back | steps |
|---|---|---|---|---|---|---|
| 1 | code |  | tables | `{}` | tables-bf21a9 | - |
| 4 | model | analyst f1 | gap | `{"a":"female","across":"state","b":"male","field":"gender","table":"lfpr_annual","where":{` | gap-b07602 | S2 |
| 5 | model | analyst f2 | gap | `{"a":"female","across":"state","b":"male","field":"gender","table":"ur_annual","where":{"a` | gap-0d7360 | S1 |
| 7 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': 'female', 'across': 'state', 'b': 'male', 'field': 'gend` | recheck-0cb81d | - |
| 9 | code | ask_analysts | recheck | `{'analysis': 'gap', 'args': {'a': 'female', 'across': 'state', 'b': 'male', 'field': 'gend` | recheck-9c1ca7 | - |

### Q4#1: What is the unemployment rate of Muslim women in Kerala?

Answer: The PLFS does not publish unemployment rates specifically for Muslim women in Kerala. The closest available figures are: the unemployment rate for Muslim women across all states was 3.6% in 2025 (PS+SS measure), and the unemployment rate for women in Kerala (all religions) was 6.4% in 2025 (PS+SS measure).

The lead's tasks:
- (call 3) Get UR for religion=Muslim, gender=Female (both urban and rural combined) from the annual 2025 release.
- (call 3) Get UR for state=Kerala, gender=Female (both urban and rural combined) from the annual 2025 release.

| n | by | who | analysis | arguments | came back | steps |
|---|---|---|---|---|---|---|
| 1 | code |  | tables | `{}` | tables-bf21a9 | - |
| 4 | model | analyst f1 | value | `{"table":"ur_annual","where":{"gender":"female","period":"2025","religion":"Islam","sector` | value-ad6eb4 | S2 |
| 5 | model | analyst f2 | value | `{"table":"ur_annual","where":{"gender":"female","period":"2025","sector":"rural + urban","` | value-681828 | - |
| 7 | code | ask_analysts | recheck | `{'analysis': 'value', 'args': {'table': 'ur_annual', 'where': {'gender': 'female', 'period` | recheck-d8a8c9 | - |
| 9 | code | ask_analysts | recheck | `{'analysis': 'value', 'args': {'table': 'ur_annual', 'where': {'gender': 'female', 'period` | recheck-1adf28 | - |
| 11 | code | json | value | `{'table': 'ur_annual', 'where': {'gender': 'female', 'religion': 'Islam', 'sector': 'rural` | refused: ur_annual: gender x religion x state is not published together. Published together: education x gend | S1 |

### Q8#1: Does getting a degree make people more likely to be unemployed?

Answer: In 2025 the unemployment rate for graduates (education level 7) was 11.2%, while the overall unemployment rate for all ages, All India was 3.1%. In 2024 the graduate unemployment rate was 12.9% and the overall rate was 3.2%. The data show that graduates consistently had a higher unemployment rate than the overall population, and the difference between the two rates narrowed from 2024 to 2025.

The lead's tasks:
- (call 3) Find the unemployment rate (UR) for education levels 7.graduate and 8.post graduate & above and the overall UR for all ages, state All India, measure CWS, for years 2024 and 2025; give the gap between each education level and overall and whether the gap widened or narrowed.
- (call 3) Find the unemployment rate (UR) for education level 6.secondary and 5.upper secondary, to compare with degree holders, same specs as above.

| n | by | who | analysis | arguments | came back | steps |
|---|---|---|---|---|---|---|
| 1 | code |  | tables | `{}` | tables-bf21a9 | - |
| 4 | model | analyst f1 | gap | `{"a":"7.graduate","b":"all","field":"education","table":"ur_annual","where":{"measure":"CW` | gap-e48e18 (wasted) | - |
| 5 | model | analyst f2 | value | `{"table":"ur_annual","where":{"education":["higher secondary","diploma/ certificate course` | value-760a28 | - |
| 6 | model | analyst f1 | gap | `{"a":"8.post graduate \u0026 above","b":"all","field":"education","table":"ur_annual","whe` | gap-866baf (wasted) | - |
| 8 | model | analyst f1 | gap | `{"a":"7.graduate","b":"all","field":"education","table":"ur_annual","where":{"measure":"CW` | gap-c14e2c (wasted) | - |
| 10 | code | ask_analysts | recheck | `{'analysis': 'value', 'args': {'table': 'ur_annual', 'where': {'education': ['higher secon` | recheck-914186 | - |
| 11 | model | analyst f1 | gap | `{"a":"8.post graduate \u0026 above","b":"all","field":"education","table":"ur_annual","whe` | gap-52f7d9 (wasted) | - |
