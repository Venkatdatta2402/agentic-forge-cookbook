# Stats Desk: design

> An AI analyst for India's official statistics. When MoSPI publishes new data, code builds a
> **profile** of it: where it's high, where it's low, which gaps change across which groups,
> what moves together, what changed. When someone asks a question, a team of agents reads the
> profiles, picks the patterns that bear on it, forms hypotheses, tests them with drill-downs,
> has a skeptic try to break each finding, and writes a brief, with every number traceable to
> a published MoSPI figure.

Status: **profiles built and tested; investigation layer being rebuilt on top of them.**
Section 12 says what exists and what doesn't. Numbers marked *measured* come from runs on the
live services; numbers marked *estimate* do not yet.

---

## 1. The idea, as one trace

```
MoSPI publishes PLFS for a new month/year
  └─ code (no AI): fetch every published table → PROFILE (stats, interactions, correlations,
                    changes, data checks) → ~600 tokens per indicator for the agents

A user asks: "Why are young women in cities struggling to find work?"
  LEAD reads the unemployment and participation profiles and picks what bears on it:
     · urban youth unemployment: women 18.9%, men 11.8% (2025, usual status)
     · the women-minus-men gap is +7.1 in cities, +0.4 in villages      (an interaction)
     · women's participation 40.0% vs men's 79.1%
  LEAD forms hypotheses, e.g.
     H1 "the gap is about education: more young urban women are graduates, and graduates
         have higher unemployment"  → drill: unemployment by education x gender, urban
     H2 "it's seasonal or a one-year blip"  → check the monthly series and 2024
  ANALYSTS carry out the lead's tasks, several at once when independent (code computes)
  SKEPTIC tries to break each finding: another measure? a small group? true by definition?
          a cause read into a correlation?
  LEAD writes the BRIEF: findings with their evidence tables, what was ruled out, what the
       data can't tell
```

The figures in this trace are real (profiles of 2025, measured). The hypotheses are examples
of what the lead would form; their answers are not known until it runs.

**Who decides what:**

| Who | Decides |
|---|---|
| Code (profiles) | nothing: computes every published combination the same way for every release |
| Lead | which patterns matter for this question, which hypotheses to test, in what order, when to stop |
| Analysts | how to carry out the lead's task: which tables, which splits, which analysis |
| Skeptic | what could make a finding wrong, and whether it survives |
| Code (gate) | whether each number in the brief is backed by a published figure |

---

## 2. The data: what we verified about MoSPI

All verified with live calls (2026-11-07 to 11-09), mostly on PLFS (the labour force survey).

| Fact | Evidence |
|---|---|
| MoSPI's official MCP server `mcp.mospi.gov.in/mcp`: no sign-in, 4 tools, 31 datasets | connected with the repo's MCP client |
| Rows are **published estimates**, not records: no IDs, one row per combination of fields | every row unique by its fields; 0 duplicates |
| Filters **select stored rows**; nothing is computed per request | leaving gender out returns male, female AND person; unpublished combinations (monthly by state) don't exist |
| Monthly PLFS is a **complete grid** (month x age x gender x sector) | 216 of 216 combinations |
| Annual PLFS is **several published tables**, 0.4% of the full grid | religion and social group only at All India, never with each other or education |
| Totals are rows: `person`, `rural + urban`, `All India`, `all` (empty = `all`) | each lies between its parts in **100%** of groups (e.g. gender 1,598/1,598) |
| **No total**: weekly status (CWS vs PS+SS are two measures), year, month | CWS >= PS+SS in 638 of 660 pairs: different measures, never mixed |
| Age bands overlap (15-29 inside 15+ inside All ages) | 15+ and All ages differ by <= 0.2 points for UR |
| Education tables leave age empty; they cover **15+** | their total equals the 15+ figure (LFPR 59.3 = 15+, not 44.9 All ages) |
| Monthly PLFS uses the **weekly measure (CWS)** | 2025 monthly UR averages 5.2%; annual CWS 5.3%, PS+SS 3.1% |
| MoSPI's definition text is unreliable | the UR definition is a copy of the LFPR definition |
| Rates can't be averaged into totals | women 26.6 + men 16.4 don't average to the real 19.0 (urban youth, Sept 2026) |
| Paging: 100 rows a page works, 1,000 is refused | live calls |
| **Rate limit, unpublished**: 429s after ~150 quick page requests | profile fetch failed on page 42 of its 3rd table; paced at 0.4 s/page it ran clean |
| Errors arrive as success (`is_error=False`, an `"error"` field inside) | live calls |
| Raw responses are large: list_datasets 5.7k tokens, CPI metadata 29.5k | measured |

---

## 3. Layer 1: the profile (code only, per release)

### 3.1 Fetching

**One unfiltered fetch per indicator per period.** MoSPI returns every published combination,
from the finest cells to the All-India total, so every split (one field, two fields, any mix)
is then a lookup on our side, not another fetch.

| | Rows | Pages | Time (*measured*) |
|---|---|---|---|
| PLFS annual, one indicator, one year | 5,490 | ~55 | ~40 s |
| PLFS monthly, one indicator, one year | 216-243 | 3 | ~1 s |
| A full PLFS release (3 indicators; this and last period) | ~34,000 | ~340 | ~4 min |

Paced one page at a time (0.4 s apart); a 429 is waited out (5, 15, 45 s) and never cached.

### 3.2 Order of the build, and reading a figure

1. **Checks on the raw download** (`release/cells.py`): every total lies between its parts, the
   shape is the known one (the guard), which age band the age-less tables cover, which measure
   the monthly series uses.
2. **The stored tables and their cards** (`release/store.py`).
3. **The profiles, from the stored tables** (`release/build.py`), with the same calculations
   the agents' tools use (`analysis/compute.py`): the profile is those analyses run at the
   default settings, with the most striking results kept. One copy of each rule, so a profile
   and a drill-down can't disagree (decided 2026-11-10; before, the profile had its own copy,
   and a rule fixed in one place had to be fixed in the other).

Reading a figure: a request names some fields; every other field with a total is at its total,
and MoSPI's own row for that combination is returned. Nothing is computed from other rows.

### 3.3 What a profile contains

| Part | What | How |
|---|---|---|
| headline | MoSPI's total; on both measures where both exist | lookup |
| one-way | every group of every field with a total, vs the total | lookups; spread, median, outliers |
| interactions | how a gap (women-men, urban-rural, youth-all) changes across another field | differences of published cells; ranked without statistical outliers, which are named |
| correlations across states | within an indicator (women vs men...) and between indicators | ordinary and rank side by side; when they disagree, the states driving it are named |
| changes | vs the previous release | lookups in both |
| monthly | series, gaps in it, vs last month, vs same month last year, swings | lookups |
| monthly, over time (2026-11-13) | the series' trend, latest streak, latest month against its own history; groups with a clear trend or a 3+ month streak | the `trend` and `series` tools' own rules (`compute.trend_verdict`, `streak`, `against_history`), so the profile and an analyst never disagree |
| striking | top 3 of each kind, ranked within the kind by its own measure | code |

Not in profiles (left to the lead's hypotheses and the analysts): `average`, `drill`, the
3-month averages, and anything below the top level (e.g. gaps among young people in villages).

The data checks (totals between parts, copies that disagree, the measure inferred) are in each
table's card, where they describe the table, not in the profile.

What agents read is a text version, **~550-650 tokens per indicator** (*measured*); the monthly
ones ~730-780 since the over-time lines (2026-11-13).

### 3.4 Rules: what's generic and what's written for PLFS

| | Generic, re-checked every build | Written for PLFS |
|---|---|---|
| lookups, outliers, correlations, ranking | ✓ | |
| monthly measure | inferred from the numbers each build | |
| age range of empty-age tables | inferred from the numbers each build (must match in every indicator) | |
| which values are totals | the between-parts test runs every build | candidate names (`person`, `rural + urban`...) |
| which gaps to compute | | women-men, urban-rural, youth-all |
| correlations that are true by definition | | WPR = LFPR x (1 - UR/100); 15-29 inside 15+ |

**The guard:** if a total fails the between-parts test, an unknown field appears, or a field
loses its total row, the build **stops and reports what changed**; the previous profile stays in
use. A release that changes shape can't produce a quietly wrong profile.

**"Written for PLFS" now lives in one file: `src/stats_desk/datasets/plfs.toml`** (Phase 1 of
the dataset redesign, 2026-11-25). Before, PLFS facts were spread through the code (column
names, totals, defaults, the measure's names, which indicator a task means, the true-by-definition
identity, that `relate` works across states) and the prompts (the lead's opening line, the
examples, "only rates are published"). Now the code and the instructions take every one of them
from the description, as in a semantic layer (SDMX's data structure definitions; dbt's,
Cube's or Cortex Analyst's semantic models): dimensions with their totals, defaults, names and
synonyms; the indicators and the words that name them; identities; how MoSPI's columns map onto
ours; the profiles' contrasts; and the examples the instructions show. What is published together
is still worked out from the stored rows, as SDMX's "actual constraints" are. Tool descriptions
are templates (`$across_plural`, `$ex_gap`...) filled in when the server registers them.

Checked to change nothing: every prompt, tool description and schema, the tables list, the
profiles, the release rebuilt from data/raw (every row and card), 123 saved analysis calls
replayed and the task check on 174 saved tasks: 512 entries, byte-identical before and after.

Still PLFS-shaped, for Phase 3 (a second dataset) to settle: exactly two settings, a measure and
an overlapping band (`role` in the file); one dimension to correlate across; rates in %; annual
and monthly tables; MoSPI's own per-dataset selectors in `release/mospi.py`; the web page's text.

**Phase 2: what each call carries no longer grows with every value and table** (2026-11-25).
Tokens per call, measured with no model call:

| | before | after |
|---|---|---|
| lead's list of profiles (every lead call) | 683 | 479 |
| analyst's tables list (every analyst call) | 709 | 615 |

- **Long value lists are matched, not shown.** A field with more than 12 values (the 37 states;
  CPI's items later) is given to analysts as a count. The server matches what they write
  (`Table._check`): the description's synonyms (a religion's followers for the religion, "women"
  for female, "cities" for urban, old state names), "and" for "&", a misspelling close to one
  value only; on a miss the refusal lists the closest values first. This is how semantic layers
  handle fields with many values (Cortex Analyst's synonyms and value search). The lead's task
  check matches the same synonyms, so a task naming "Muslim women" on CWS is now caught; before,
  only "Islam" was. Span tests (2 each): an analyst asked about "Muslim" and "Hindu" women, and
  one about "Jammu and Kashmir", "Tamil Nadu" and "Orissa", each handed in on the right groups
  (4 of 4), with no refused name.
- **What is published together is said once for tables split alike** (the three annual tables
  repeated it, ~165 tokens each), and only for tables with limits. Two tries that didn't work,
  both span-tested. Moving it into the profile replies: the lead read all six profiles, Q8 in 3
  of 4 picks while the list said it would come with them, and Q1 in 2 of 2 after that line went
  (2 of 2 read only the annual three on the old list): it couldn't tell which table held the
  question's groups together. Saying of the monthly tables that every field goes with every
  other made them look the fullest: Q1 read all six, 2 of 2. As built: pick Q1 and Q8 without
  all six, 4 of 4; tasks Q4 and retask R1 pass code's task check, 4 of 4 (tasks Q1 and Q8, 4 of
  4, on the list before its monthly entry went).
- No lookup tools were added: each use would be a model call (~3k tokens), more than the lists
  it replaces.
- **A task asking for a group whose names aren't published together is refused** (Q4's lead put
  Muslim women and Kerala in one task in 1 of 4 span plans on 2026-11-25, and in 1 of 5 eval
  runs; given it, an analyst said so, 2 of 2, at ~11k tokens, and in the first eval passed off the
  all-India figure as Kerala's). Code finds the names (`names_in`; a name inside a longer one
  doesn't count, so "post graduate" is not also "graduate"), but not how the sentence puts them
  together: "Muslim women in Kerala" is one group, "Muslim women against women in Kerala" two.
  One small call (`agents/grouping.py`, gpt-oss-20b, ~430 tokens) reads that, only for tasks
  naming two fields or more (61 of the 207 tasks leads have written); code then asks the server
  whether each group is published (no tokens). A reply it can't read, or a failed call, lets the
  task through: a misread must never block a valid task.
  Measured on every such task leads wrote, labelled by hand, plus 6 written to set the same
  names together and apart (`evals/grouping.py`, 67 cases): what the reply looks like mattered
  most. A list of names was read two ways (one group, or one field's names): 117 and then 106 of
  134 right, 2 valid tasks refused. Each group as an object, one value per field from the names
  found: 61 of 67 right, none refused wrongly, none missed; with the groups read past a stray
  quote that made Groq refuse 4 replies, 62 of 67 on the same replies scored offline. The rest
  give fewer combinations than meant ("women and men in Nagaland" as women, men, Nagaland), never
  a wrong one. Retask span R2 (the lead's plan with that task, refused): the lead gave published
  groups instead, 2 of 2.
  Which names go together is left to the model: a rule in code joining names the reply left
  apart (female, male and Nagaland each alone as female and male in Nagaland) fixed the 2 saved
  replies it applied to, but it reads the reply's shape, not the sentence, and with several
  fields split at once it would guess, and could refuse a valid task. Dropped.
- **The lead is told up front which of the question's groups aren't published** (`Investigation.
  read_question`): the same call, on the question, before the lead's first call; for a group
  that isn't published together, code finds the nearest that are (one name left out at a time,
  asked of the server), and the lead gets one note with the question. Nothing is said when every
  group is published. The question's words are the asker's meaning; the task check stays, on
  each task, as what the analyst will act on. Its refusal is also the evidence a brief needs to
  say the combination isn't published.
- **The call runs on gpt-oss-120b.** 15 harder cases added (H: several fields split at once,
  several values of one field, names that apply to some groups only; 4 not published), 82 in all.
  120b, 2 runs: 162 of 164 groups right, every hard case (30 of 30), no valid task refused, none
  missed, no reply refused by Groq, ~440 tokens a call; its 2 misses are tasks whose label is
  arguable ("and others"; "by gender (male and female combined)"). 20b at low effort had had
  ~85-90% on the 67 easier cases, up to 2 valid tasks refused. 20b at medium effort was not run.
  On the same model as the agents, a free question's calls may be spread over the desk's free
  keys by room in their minute (QuestionKeys.pick), as the agents' are.
- Span tests with the up-front note (2026-11-26): tasks Q4, 4 of 4 asked for the nearest
  published groups (women in Kerala; Muslim women in all of India), none for the combination
  (1 of 4 the day before, without the note), none refused by code. Pick and tasks on Q1 and Q8,
  2 each: as before (needed profiles, not all six; tasks pass code's checks). Tokens: ~370 for
  the question's call, ~480 for each task naming two fields or more (Q1's plan: ~1.3k more).
- An analyst's hand-in cites result ids read by their shape (`Analyses.cited_ids`), however
  written: in saved runs all 12 ids refused as unknown were real ones written as "result
  value-760a28", 8 of them two to an entry. A made-up id is still refused. The lead's brief cites
  finding ids the same strict way as before: its 9 refusals in saved runs were real mistakes
  (a profile cited as a finding, a finding that didn't exist).

### 3.5 Honesty rules built into the text

- Correlations across states describe **states**: people's chances there, not why. "A woman
  looking for work in a high-participation state is more likely to be unemployed" is a fair
  reading; "participation causes unemployment" is not.
- Correlations true by definition are listed as **not findings**.
- Across states, statistical outliers (the small territories, surveyed with small samples) are
  set apart and named, never headlined. In other fields a far-off group is a real, large group
  and a finding, so it stays in: setting outliers apart everywhere had ranked "women minus men
  across education" without its graduate groups (+8.4, +10.6) and dropped it from the profile.
- Every figure names its measure; the two weekly-status measures are never mixed.

### 3.6 The release job (built 2026-11-14)

**GitHub Actions, daily at 19:00 India time, free, no tokens**
(`.github/workflows/stats-desk-release.yml`, `scripts/update_release.py`, `release/update.py`):

- **The raw downloads live in the repository** (`data/raw/plfs`, 11.4 MB; user's choice
  2026-11-14), so the job fetches only what's new and builds from the rest. A new month changes
  three ~50 KB monthly files; a new annual year adds ~6 MB once a year.
- **The check: three small requests**, measured 2026-11-14: the annual years and the monthly
  years MoSPI lists in its metadata (newest first: a year newer than ours is new, whatever
  MoSPI calls it), and the row count of our newest monthly year (more rows = a new month;
  the metadata's months are pooled across years, so they can't show it). "No Data Found" means
  not published; an outage raises, never reads as "nothing new".
- **What's new is fetched for all three indicators**: a new month = that year's monthly table
  again (MoSPI serves a year as one table); a new year = its tables for the first time.
- **Revisions: MoSPI's figure wins, and it's logged** (user, 2026-11-14: MoSPI's figures are
  the official ones). MoSPI publishes no revision policy we could find; re-downloading the 2025
  and 2026 monthly unemployment tables on 2026-11-14 matched all 459 rows saved on 11-09
  (nothing had been released in between, so that proves little). A re-fetched table replaces
  ours whole; every figure it changed or dropped goes to `data/raw/REVISIONS.md` (date, table,
  row, old -> new) and is counted in the commit message. The release label now ends with a
  fingerprint of every figure ("... monthly to 2026-09 · data 311908"; the same on every build
  of the same figures), so a correction changes it even with no new period, and a follow-up to
  an older brief gets the "re-released, don't mix" warning. Revisions are seen when a table is
  fetched again: a year's monthly table with each new month; annual tables only when their year
  first appears.
- Then: rebuild (the build reads its years from `data/raw`, so a new year needs no code
  change; same profiles byte for byte when nothing changed), offline tests, commit the data as
  github-actions[bot], and start the deploy workflow (a GITHUB_TOKEN push doesn't trigger it).
- GitHub pauses scheduled workflows in a repository with no activity for 60 days.

Still to check: MoSPI's data licence.

---

## 4. Layer 2: the investigation (agents, per question)

### 4.1 The team

| Agent | Model | Job |
|---|---|---|
| **Lead** (principal analyst) | gpt-oss-120b | reads the profiles relevant to the question, finds the patterns that bear on it, forms hypotheses, directs the analysts, decides whether to go on and how, writes the brief. **Does not run analyses itself** |
| **Analyst** (one general kind, several instances) | gpt-oss-120b | carries out one task the lead directs (e.g. "test whether the urban gender gap holds for graduates"), using the analysis MCP server; reports back with a form |
| ~~Skeptic~~ | -- | removed 2026-11-11: code's checks (`agents/checks.py`) rerun every cited analysis in the other period and on the other measure and add warnings (unlike groups compared, outlier states, true by definition, cause words); the lead reads them with the finding and judges it, since it decides what to do next anyway |

No subject-specialist analysts (decided 2026-11-09): with every table stored and described by
code (section 5), what made analysts different -- their own datasets, codes and notebooks -- is
gone. One analyst kind does all tasks. All on 120b: on day 1 gpt-oss-20b failed the analyst
workflow **4 of 4**; 120b followed it **5 of 5** (*measured*).

### 4.2 Orchestration

- **The lead directs, the analysts execute** (chapter 6's supervisor with agents as tools):
  `ask_analysts(tasks)` is a tool of the lead's; each task starts an analyst instance that sees
  only its task (chapter 2: a branch gets its goal, not the parent's conversation). The lead
  says what to find out, not how: the analyst chooses the analysis.
- **Independent tasks run in parallel, by code**: `ask_analysts` takes a list and runs one
  analyst per task at the same time. First it was `ask_analyst(task)`, with the lead told to
  call it several times in one turn; on live runs 1 and 2 gpt-oss-120b made one call per turn
  both times, so parallelism can't be left to the model's choice. `profile` likewise takes
  several names.
- **Parallel, within one budget**: all instances spend from the asking user's Groq key (8,000
  tokens a minute on the free tier), so two at once each run at about half speed. The gain is mostly small, separate contexts, not
  time. Each instance gets a share of the question's token budget (chapter 2's reserved budget).
- **The skeptic is fired by code** on every finding, not left to the lead's choice (chapter 6:
  agents skip optional steps): each task runs the analyst, then the skeptic on its finding, and
  the lead gets the finding with its verdict.
- **Code gathers the skeptic's evidence; the skeptic judges** (decided 2026-11-10). On live runs
  1-2 a skeptic with tools re-fetched figures it had been shown or re-ran the analyst's own
  analysis in other words; it caught nothing (it missed run 1's comparison of young urban women
  with all women) and cost ~45% of a question's tokens. Its checks are mostly mechanical, so
  code runs them: `recheck` reruns each cited analysis in the other period and on the other
  measure, and `flags` names figures compared that differ in more than one field, outlier states
  named in the finding, correlations true by definition, and cause words. The skeptic reads it
  all and gives its verdict in one call with no tools.
- **Limits** (built, `agents/lead.py`): at most 4 analyst tasks per question; one token budget
  for the whole question (60k), because the lead's runtime counts every token spent under it,
  its analysts' and skeptics' included (the meter follows the task); `ask_analysts` refuses
  tasks the budget can no longer pay for, keeping enough for the brief. Per agent: analyst 12k,
  skeptic 6k. Chapter 2's loop detection. A limit stops the next model call, never the carrying
  out of a decision already made (live run 2 lost a skeptic's verdict that way).
- **What each agent sees** (chapter 5): the lead gets the question and the list of profiles and
  reads the profiles it picks; an analyst gets its task and the tables list with every field's
  exact values (620 tokens: a misspelt group costs a refused call); the skeptic gets the finding,
  the results it cites, code's reruns and flags. When to use which analysis is in each tool's
  description, where the model reads it when choosing, not in the agents' instructions.

### 4.3 How the agents compute: an analysis MCP server, run by code

Agents choose **what** to compute; code computes it. The analyses are our own **MCP server**
(chapter 7), read-only over the release file; the analysts and the skeptic connect to it.
MoSPI's MCP server is used only by the release job. Built (`src/stats_desk/analysis/`):

| Tool | What it answers | Example |
|---|---|---|
| `tables` | what's stored; with a name, that table's card | |
| `profile` | a profile's text | |
| `value` | one published figure | urban women 15-29, UR 2025: 18.9 |
| `compare` | every group of a field vs the total, ranked; spread; outliers (states only) | UR by education: graduates 11.2 highest |
| `gap` | a minus b, overall and across another field's groups; where it reverses | women minus men across sector: urban +7.1, rural +0.4 |
| `change` | vs the previous year; monthly: vs the previous month and a year earlier; per group with `by` | UR 2024 3.2 -> 2025 3.1 |
| `series` | one figure over every period; months not published; low, high, largest move | |
| `average` | one figure averaged over a stretch of months, optionally against another stretch; labelled "an average of the monthly rates, not a published figure" | UR Jan-Jun 2026 5.22 vs Jul-Sep 5.03 |
| `trend` | the straight line through one figure's months, against the usual scatter around it: no clear trend / slightly rising / rising; never "significant" (MoSPI gives no margins of error) | UR 2025-04 to 2026-09: -0.2 along the line, scatter 0.3: no clear trend |
| `relate` | across states: ordinary and rank correlation, the states driving a disagreement, BUILT IN when true by definition | LFPR vs WPR r = 0.99: built in |
| `drill` | where to split next: each unset field, how much its groups differ, widest first | UR: education 10.9 points, state 9.1, sector 2.4 |
| `recheck` | the same analysis in the other period and on the other measure; run by code for the skeptic, not offered to analysts | UR headline: 2024 3.2, CWS 5.3 |

How a request reads the data: it names some fields in `where`; every other field with a total is
at its total; age (15+), measure (PS+SS) and period (the latest) get a stated default. Every
result says which figures it is ("UR, annual, 2025 | usual status (PS+SS) | age 15 years and
above | gender = female"). An unpublished combination is refused with what *is* published
("religion x state is not published together. Published together: education x gender x sector
x state; gender x religion x sector; ..."); a wrong value with the valid ones.

Rules in every analysis, in code: only fields with a total are left out of a request; never
average or add rates; groups that contain others ("9.secondary & above") and overlapping fields
(age bands, the two measures) shown but never ranked; months not published are named, never
filled; correlations say they describe states, not causes. **No free SQL** (decided 2026-11-09):
what the menu can't compute, the agent says it can't answer yet, and the menu gets a new analysis.
Not built, because the stored data can't support them yet: real terms (needs wages and CPI),
trends over years (2 annual periods until history goes back to 2017-18), shares (need head
counts; MoSPI publishes rates). Averaging (changed 2026-11-12, user's call after the span tests
showed the lead asking for moving averages): never across groups -- women's and men's rates don't
average to everyone's -- but over months of one figure, labelled, since the labour force barely
changes size month to month. `series` also gives the latest 3-month average against the 3
months before. The two new tools cost the analysts' tool schemas 1,237 -> 1,620 tokens a call.

How a call ends is kept apart on the wire (`_meta` kind): answered; **refused** (the message
says what to do); **broken** (a bug: logged, and kept out of the conversation). Arguments are
checked on our side against the server's schema, because the server silently drops an argument
it doesn't know (measured: `wher` instead of `where` was answered as if absent).

One server process per question, shared by all its agents; tool schemas cost **1,068 tokens**
per call (*measured*; the defaults are said once in `tables`, not in each parameter, which saved
280).

### 4.3b What changed after the first eval (2026-11-11)

The 5 baseline runs ended with no brief 3 times. In order of damage, and what was done:
- gpt-oss called tools named "json" or "commentary" 9 times, with a complete hand-in as the
  arguments; Groq rejected each. The hand-in tool is now named `json`, and a rejected call whose
  arguments fit exactly one tool is made as that tool (`Think._recover`); a hand-in written as
  text is read as the form, with the same checks (3 tries).
- The lead read all 6 profiles, then the budget couldn't pay for its tasks. It now first sees the
  profiles by name and table only, and once findings arrive the profiles shrink to two lines.
  A budget refusal says how many tasks it can pay for.
- An empty brief was accepted; it is now refused.
- The hand-in is one sentence plus result ids, and the lead reads the figures written short by
  code from the results (`analyst.compact`), the reruns and the warnings.
- The brief's numbers must be in the findings' results or code's reruns, not the profiles.
- First-call prompts, *estimated*: lead 1,309 tokens (profile list 322), analyst 2,492.

### 4.3c What changed after the first live question on the page (2026-11-17)

"Why is unemployment so high among educated young people in India?" ran on the real app
(2026-11-14): 49k tokens, 5.5 minutes, no brief. The lead gave 4 tasks at once; two analysts
asked for education "graduate" (MoSPI writes "7.graduate"), were refused, repeated it and were
stopped; one ran out of budget looking figures up one by one; the lead then wrote that
unemployment by education isn't published (its profile showed 11.2% for graduates), as text,
three times, refused each time for a "29" that was only part of "ages 15-29". Fixed:

- **Group names** (`analysis/data.py`): a value matching a published one without its number
  ("graduate" -> "7.graduate") is that one; one that could still be several is refused naming them.
- **Age groups aren't figures** (`analysis/client.py` AGE): "ages 15-29", "15+", "aged 15 and
  above" are left out of the number check.
- **A task that came back empty** (`lead._unfinished`): says it proves nothing about what is
  published, passes on what the server refused (which does), and that the task can be given again.
- **The lead plans** (`agents/plan.py`, user's request): `ask_analysts(approach, tasks, later)`:
  its approach in a sentence or two, the tasks that need no other task's answer (at most 3 in
  the first call, FIRST_ROUND, keeping one to follow up with), and later tasks that wait for an
  answer, each saying which. The later tasks come back to it with the findings, to give as the
  findings make them look or drop. How to plan -- explore as an analyst would (where it is high
  or low and for whom, how it has moved, what moves with it; for a "why", where the problem
  concentrates first, then what goes with it there; tasks that tell competing explanations
  apart) -- is in the plan's field descriptions, read while writing it: in the instructions, the
  same words had the lead skip planning and write its brief from the profiles in 5 of 6 span
  runs (2026-11-18). From a longer prompt the user brought, the instructions took six short
  rules: an overall figure can hide the opposite within groups; the nearest published figures
  stand in for an unpublished combination, they don't answer it; never piece a combination
  together from two published ones; stop when more tasks are unlikely to change the answer;
  keep levels apart from changes. Not taken: its length (~2,500 tokens a call against ~900),
  rules code already enforces, and a JSON-only answer that would break the brief form.
- **The lead's first steps are fixed by code** (`Investigation._next_step`, the API's
  `tool_choice` naming the tool): read profiles, then ask_analysts, before it may hand in; free
  after the first finding, and in a follow-up whose earlier findings can be cited. Where a
  profile already shows the answer (Q2 always, Q5 mostly), the lead wrote its brief from it in 8
  of 12 span runs whatever the instructions said. Groq checks a named call after generating it,
  so a model set on the brief is refused, not steered: the tool is named for one try, the call
  is never rescued as another tool (it was, as the brief), and the next try is the model's own.
  Tasks name groups
  as the profiles write them; the list of profiles says, per table, which fields MoSPI publishes
  together (`Table.combinations`, by code: age groups 15-29 and 15-59 only by gender, sector,
  state; education only for ages 15+; religion and social group never by state). Span tests
  (E1): groups named as published 3/3 (0/3 live); the unpublished combination asked for in 1 of
  3 (2 of 3 without the list).
- **Tried and dropped the same day: tasks as structured fields** checked by code against the
  published combinations, with every allowed group listed in the tool's definition. gpt-oss-120b
  invented field names ("id", "action", "measure1"); Groq refused 5 calls against the schema
  (~4k tokens each); a call that fitted wrote the groups in the sentence and left their fields
  empty, so the check checked nothing. Listing the allowed values in the analysts' own tools was
  measured, not built: +2,700 to +4,700 tokens on every analyst call (the 10 tools' definitions
  are 1,711 now), too much for a key with 8,000 a minute. Also tried: hiding the brief form until
  a finding was in -- the model called it anyway, Groq refused it as "not in request.tools", and
  after 4 tries the turn failed.
- **One answer per group** (user: "two lines on a chart" was an example, for every analysis
  where it means something): `series`, `trend`, `average` and `relate` take `by` -- one line,
  average or correlation per group of a field side by side, and for two groups their gap and how
  it moved (or whether the lines crossed). `compare`, `change` and `gap`'s `across` already did.
- **Groups are chosen the same way in every analysis** (said once, in the rule the tables list
  ends with): `by` a field is every group of it and the total; a list for that field in `where`
  is only those groups and the total; several fields set in `where` pick one combination; `where`
  a list of settings is several groups each set on its own (e.g. rural women in one state, and
  urban women), with the total on what they share. `value`, `change`, `series`, `trend` and
  `average` put them side by side (for two, their difference, saying when they differ in more
  than one field: not like with like); `compare`, `gap`, `drill` and `relate` run within each. `gap`
  takes several groups against one (`a` a list, or `"*"` for every group), each with its earlier
  gaps. Before, `gap` took two groups only: an eval lead asked for graduates and post-graduates
  against everyone in two years, and its analyst needed two gaps and two lookups.
- **Every agent turn is a tool call** (Think `require_tool`: the API's `tool_choice: "required"`):
  told to "just write the answer in plain text", gpt-oss-120b called the tool 3 of 3 times
  (measured). Reading a text reply as the form stays as the backstop. Brief spans Q1, Q2: handed
  in at the first try.
- **Free questions borrow the idle free key** (`llm.QuestionKeys.pick`, user's request): each call
  goes on whichever free key has more room in its minute, while no other question holds the
  other key; checked every 20 s, so a key another question takes is given back within that.

**Two full questions on the real app (2026-11-18)**, "In cities, is the gap between women's and
men's unemployment narrowing or widening?". First run: the lead took "cities" to be unpublished,
answered for all India and said urban-by-gender isn't published (it is); an analyst wrote prose
three times and spent its budget on no analysis. Fixed, and the second run asked for urban
figures, read both profiles, used the two-line `series`, and gave a failed task out again:
- the tables list says what groups mean (`MEANINGS`: urban = towns and cities; no single city);
- a brief saying something "isn't published" is refused unless an analysis in this question was
  refused as unpublished (`_refused_as_unpublished`);
- both prompts: every reply is a tool call, never prose; an analyst runs the closest analysis
  when none does all of a task; the lead never hands in before analysts have reported. Prose
  that slips through gets its own words back with that next move (`when_stuck`), not Groq's
  "issue the call again", which had the analyst repeat itself;
- `gap` gives the same gap a period earlier (and a year earlier, monthly): widened, narrowed or
  reversed, in one analysis;
- two lines over time (`trend` with `by`) say whether their gap is widening or narrowing from
  the straight line through every month against its usual scatter, never from two months
  (`series` describes and gives no verdict, user: series describes, trend judges): the second run's brief called a gap "widening" that swung between 2.1 and 3.8
  points (its line: +0.01 a month against 0.5 of scatter, no clear trend).

**Eval rerun (2026-11-18, `evals/reports/runs_2026-11-18_after_fixes.md`)** against the 11 Nov
baseline, Q1-Q5: briefs 5/5 (was 2/5), points 6/11 (2/11), measure named 5/5 (1/5), wasted
analyses 1 (7), tasks stopped 1 (3), tokens per delivered brief ~33k (~78k); Groq errors 13 (7),
mostly tool calls written as text and recovered (the report now counts those apart). Q6-Q8: 3/3
briefs. Its misses, each traced in the saved runs and fixed:
- Q5 (true by definition): barred from citing the flagged correlation, the lead cited state
  rankings showing the same link. Now any finding flagged true by definition must be said.
- Q7 (outlier): gpt-oss wrote "Andaman &" + a narrow no-break space + "Nicobar"; checks now
  compare text with every kind of space as one (`plain`, checks `_spaced`).
- Q2 (a skipped month, and "fell"): a brief resting on monthly figures that skip a month must
  say so; a claim of rising or falling against a trend's "no clear trend" is flagged to the lead.
- Q1 (six single lookups instead of `gap`): headed off in the tool descriptions, not refused at
  hand-in (that would cost more calls, user). All descriptions were rewritten to one shape: the
  question the tool answers, what it returns, what each setting does, and what to use instead
  (series: the path month by month; trend: the direction; change: two points in time). 1,888
  tokens for the analysts' 10 tools (1,829 before).

**Eval after the second fixes (`evals/reports/runs_2026-11-19_second_fixes.md`, on the free keys).**
Q5 and Q7 now pass their traps; Q1, Q4 and Q8 gave no brief, each traced and fixed:
- the "not published" check matched honest wording ("the figures do not provide the reasons
  why") and demanded a refusal the lead couldn't give for a combination it knew from the tables
  list; refused, the lead resent the same brief until the run stopped with none. Now: only words
  that claim MoSPI doesn't publish something count; code probes the groups the question's
  analyses asked for, taken together, once (`_probe_combination`, no tokens); an unproven claim
  goes to the repair call, so it can't loop;
- told "published only for age 15 years and above", an analyst reported the figures by
  education "not published", and the brief said so, wrongly. The server now answers on the
  nearest published age group or measure, saying so in the result's last line
  (`analyses.nearest`), and "published only for…" no longer counts as evidence of "not published";
- three analysts made no model call for 15 minutes: a Groq request that never answered, waited
  out under the SDK's default 600 s a try. Requests now time out at 60 s (`REQUEST_TIMEOUT`);
- the per-group `trend` didn't list the months MoSPI skipped, so the brief couldn't be held to
  saying so; it does now. A claim saying there is no clear trend is no longer flagged for
  "rising or falling".

Rerun of the three that gave no brief: Q1 2/3 points, Q4 2/2, Q8 a brief but 0/1. Q8's analyst
made five single lookups at ~3k tokens a call, reached its 12,000 and was stopped, its figures
lost. Rather than nudge it after the waste, `value` takes lists (several groups and periods in
one call, with the field's total; at most 24 figures): the call the model reaches for, made
cheap. Code's rerun on the other measure is never moved back to the first measure by `nearest`:
that is the rerun's point.

An analyst stopped before it hands in (its budget, a loop, replies Groq keeps rejecting) has its
results handed in by code: the figures, a claim saying there is none, and a warning that it was
cut short. Tried first and dropped: forcing the hand-in as its next call once about one call of
budget was left. In a span test the model, its task not done, called another analysis instead,
Groq rejected each try, and each was paid in full: 28k tokens on a 12k budget. Stopping at a
limit is how agent frameworks end a run (LangChain's "force", the OpenAI Agents SDK's turn
limit); a claim is the one thing code can't write. Alongside: a step whose tries are all
rejected ends the agent instead of crashing the question; retries within a step stop at the
budget; Groq's "could not be parsed" (half a thought sent as the reply) gets the model's own
words back, as "did not call a tool" did. Already handled, measured on the saved runs: 5 of the
7 rejected analyst calls were rescued by the runtime (a `null` for an optional setting, a
hand-in written as text), so they cost no extra call.

Several groups and the nearest published setting (span test A7: the lead's task said "all
ages", education is published for 15 years and above only). One group refused for that is
answered by `nearest` on the published age, with a note. Several at once had each refusal
caught on its own line as "not published for these settings", so `nearest` never saw it and the
analyst reported no figures. Now, every group refused: the call is refused as for one group (and
`nearest` answers it); some: each of their lines says why. The analyst is told a note means the
figures answer the task on the nearest setting. And `value` takes lists inside a group (each
group over two years: the model's own call, refused before).

The note is put first, under the result's id, and a hand-in is checked against it. Span tests
(A7, 2 runs with the note last, 2 with it first): 4 of 4 analysts used the PS+SS figures and
never said they weren't the CWS asked; where the note sat made no difference. Now code refuses a
claim that doesn't name the setting a cited result's note gives (`check_settings`, next to the
number check): "PS+SS" or "usual status", "15 years and above" or "15+". It costs a call only
when the claim leaves it out.

Settings in the lead's tasks. 31 of the 47 distinct tasks in the saved runs named a measure or
age group, most where the question asked for neither (A7's task, from a question about degrees,
said "all ages" and "CWS"). Two causes. The lead was never told what an analyst uses when a
task names none, so it spelled settings out. And the "published together" line gave
education's age limit but no measure limits, which annual tables have on four splits:
education, religion, social group and the age groups 15-29 and 15-59 are on PS+SS only. Now that
line gives each split's measure limit too, e.g. "education: only by gender, sector, state (ages
15 years and above, measure PS+SS only)", and the lead is told to name a measure, age group or
period only when the question asks about one, with the standard ones the analysts use
otherwise.

Tasks span (Q3, Q4, Q8, twice each) after that: the lead still named a measure in 6 of 6 (12 of
12 over both span runs; the sentence telling it not to was then taken out, code does the job), mostly
the published PS+SS, which does no harm; 1 of 6 asked for graduates on "all ages" and CWS. So
code checks each task in `ask_analysts` (`unpublished_settings`): the groups a task names
(matched on the values as the tables list writes them) on the measure and age group it names,
asked of the server, no tokens, in every table of its indicator (annual or monthly, when the task
says which). A task none of them publishes is refused before any analyst starts, with the
server's nearest setting: "7.graduate (education) on measure CWS and age All ages isn't
published together in ur_annual (the nearest that is: measure PS+SS, age 15 years and above)".
Run over the 47 saved tasks: none refused. It doesn't catch a group written in other words than
the tables list's ("Muslim" for Islam); the analyst's note and claim check stay for those.

Group names, every field: "total" (or "overall") is any field's total, whatever its name
(person, rural + urban, All India, all); a value of another field put under this one is refused
saying whose it is ("'All ages' is a value of age, not of education; its total is 'all'").

### 4.4 Handing in findings

Day 1 showed a nested form (a list of findings, each with six fields) failing as a tool call
2-3 times per run. Findings are now handed in flat: the analysis results the analyst used are
referenced by id, and a short summary. Code checks that **every number in the summary is in
those results** and refuses with the missing number otherwise (built: `Analyses.unbacked`).
Ids are a hash of the request, so the same request always gets the same id, and only ids some
agent of this question was actually given can be cited. A number the analyst worked out itself
is refused on purpose: `gap`, `change` and `relate` compute the differences. Not checked yet:
that a number is cited with the right group and period ("18.9" quoted for men instead of
women); added if live runs show it happening.

The skeptic's verdict ("holds", "holds with a caveat", "does not hold") is checked the same way:
the numbers in its reason must be in the finding's results or in results it ran itself.

### 4.5 The brief

The lead hands in: the answer (2-5 sentences), the findings it rests on, what was ruled out,
what the figures can't tell. Code refuses a brief that rests on a finding that did not hold, or
quotes a number that isn't in the findings' results or the profiles the lead read. The brief
carries each finding with its verdict and the full text of every result it rests on.

Written by code, no model call: **Figures from** (2026-11-12: which table, period and measure
each figure comes from), and a **not-comparable note** in the answer itself (2026-11-14) when
the findings it rests on use different measures or mix annual and monthly figures, e.g. "Note:
the annual figures (2025, usual status (PS+SS)) and the monthly figures (2025-04 to 2026-09,
current weekly status (CWS)) are not directly comparable: different measures and periods." The
first live question had set those two against each other with "However".

**No finding ids in what the reader sees** (user, 2026-11-14): f1, e2... are for the findings
list behind the brief. Code strips "(f1)", "(f1, f2)", "[e3]" from the answer, ruled-out and
can't-tell texts once the brief has passed its checks (`lead.without_ids`), and the form tells
the lead the answer is for the reader.

---

## 5. Memory (chapter 4)

| What | Kind | Written by | Lives |
|---|---|---|---|
| **Release tables**: every published row of each stored indicator, cleaned (readable values, totals normalised, duplicates of a total merged) | facts per release | the release job | one DuckDB file per release, inside the Docker image; opened read-only |
| **Table cards**: columns, totals, fields with no total, combinations that exist, periods, measure | semantic | code, from the tables | in the DuckDB file; served by the MCP `tables` tool |
| Profiles | facts per release | code | files in the image |
| **Question records** (`agents/record.py`, 2026-11-13): one per question, everything about it: tasks, every finding (or why it stopped), every analysis result, each agent's own messages, the brief, usage | episodic | code: at the start, after each finding, at the end | Cosmos DB (in memory locally) |
| **Follow-up context**: written by code from the earlier record (below) | view of a record | code, when a follow-up starts | not stored: rebuilt from the record |

**Agents never call MoSPI** (decided 2026-11-09). Questions are answered only from the stored
tables, all from one release, so profiles and drill-downs always agree; a question outside them
is "not covered yet", and coverage grows by adding indicators to the release job. The notebooks
and "fetches that worked" (day 1) existed to help agents call MoSPI, so they go.

**Stored now:** PLFS unemployment, participation and worker share; annual 2024 and 2025,
monthly 2025 and 2026. Back to 2017-18 once everything works.

No model writes to anything a later run trusts.

**One record per question** (decided 2026-11-13; the standard shape: LangGraph's state with a
checkpointer, Google ADK's session, the OpenAI Agents SDK's sessions). Everything about a question
lives in one object, which is what is saved, and every model call's context is a view of it built
by code. Not one chat shared by all agents: each agent's own messages are kept inside the record,
since an analyst paying for the lead's chat on every call would waste tokens. The first API
version kept findings, results and transcripts in separate objects and saved a brief assembled
from some of them at the end, which left a follow-up too little to cite.

Size (*measured* on the last three saved runs, old system): 49-59k tokens SPENT per question
(every call re-sends instructions, tools and history), but only 9-12k tokens of messages and
2-6k of results WRITTEN: the record is ~15-20k tokens, ~80 KB, under Cosmos DB's 2 MB an item.

**Follow-ups** ("and what about women?" after a question about youth unemployment): code writes
the lead's opening from the earlier record (`lead.follow_up_context`):
- earlier questions in the conversation, one line each; the last brief (answer, figures from,
  ruled out, what it can't tell);
- the last question's findings as e1, e2... (this question's are f1, f2... again): task, claim,
  warnings, figures written short; the ones its brief didn't use marked so;
- their results loaded back into this question's results, so the brief can cite e1 and quote its
  numbers through the same number check, without asking analysts again. If MoSPI re-released the
  figures in between, they are not loaded and the lead is told not to mix them;
- at most 1,500 tokens: past that, unused findings lose their figures, then themselves, then the
  older questions go, then the used findings' figures.
Not shown: the stored messages (~10k tokens, where Groq takes 8k a call); kept for reading back,
evals, and a better context builder later. Analysts still see only their task.
Records live in the store (Cosmos DB), not in memory, because the host shuts an idle machine down; a
follow-up 20 minutes later must still find its question.

---

## 6. Context engineering (chapter 5)

| Mechanism | Effect (*measured*) |
|---|---|
| Profiles instead of raw tables | ~550-650 tokens per indicator, built from ~5,500 rows |
| Table cards instead of exploring a table | ~420-510 tokens per table: columns, totals, what's published |
| Rows stay in the release file; agents get analysis results | no agent ever reads raw rows (on day 1 a model shown 5 of 8 rows invented the other 3) |
| Old intermediate results shortened in the history (`shorten`); results carrying reported numbers never | day 1: prompts held at 1.8-2.4k instead of growing to 3.7k |
| Tool schemas trimmed | day 1: 837 -> 611 tokens |

Day-1 mechanisms for agents that called MoSPI (compacting its responses, notebook pages) went
with that design; their measurements are in the archive.

---

## 7. Models, limits and cost

- Groq free tier per model: **8,000 tokens a minute, 200,000 a day (rolling 24 h), 8,000 per
  request**. Groq's minute limit **refills continuously** (133 tokens/s), and Groq charges what a
  call uses, not `max_tokens` (*measured*).
- **Users bring their own Groq key** (decided 2026-11-10). Each user's limits are Groq's on
  their own account, so the app keeps no per-user or app-wide daily counts. Keys are never
  stored or logged: sent with each request over HTTPS, held in memory while that question runs,
  and checked first with Groq's model list (no tokens), so a mistyped key fails in a second.
- The governor: a refilling bucket **per key and model** (built 2026-11-12: the key in use is
  set for one question by `llm.using_key`; the governor knows it only as a hash); reservations of
  prompt + a typical reply; refused before sending if a request could never fit. It cut one
  analyst run from 425 s to 78 s. A user's bucket takes its size from Groq's
  `x-ratelimit-limit-tokens` header, so a key on a paid plan isn't held to the free tier's 8,000 a
  minute, and counts no day (Groq's 429 says when it's spent). The server's own key (scripts,
  evals) keeps the free tier's daily count.
- **A token budget per question** stays: it protects the user's own quota from a runaway loop.
  If Groq says the key's daily allowance is spent, the question stops and says so.
- Day 1 analyst runs: 16-30k tokens each (*measured*), most of it rediscovering MoSPI's codes
  and rejected forms; the release file and flat hand-ins remove both. An investigation is
  *estimated* at 30-50k tokens: **about 4-6 investigations a day per free-tier key** on 120b. To
  be measured, then this line updated.

---

## 8. Serving

- **API** (built 2026-11-12, `src/stats_desk/api/`): FastAPI, all async.
  - `POST /questions` {question, follow_up_of?} with the key in the `X-Groq-Key` header streams
    server-sent events: `started` (the brief's link, at once), `step` (one per agent event, from
    `api/progress.py`: who (lead, f1..., code, desk), what (thinking / thought with its tokens /
    waiting on Groq's minute / an analysis call and its result, refused or not / a task handed
    out / a finding in, with its figures and code's reruns / code's recheck), and a plain line
    for a reader or none: "Analyst 1 started: ...", "Analyst 1: gap on ur_annual, female against
    male (15-29 years)", "The lead is writing the brief"; finding ids stay out of the lines (user,
    2026-11-14)), then `brief`, `paused` or `failed` (said in plain words).
  - **The page** (built 2026-11-14, `api/web/index.html`, served at `/`; `/?brief=<id>` opens a
    saved brief): a question box with a short intro and the free/own-key choice (`GET /free`: the
    visitor's free questions left); the run as it happens -- the steps (profiles, tasks, analysts,
    code checks, brief), counters (time, model calls, analyses, tokens, findings), an
    orchestration map (lead, analysts, code checks, the analysis server's 12 analyses, with dots
    for each hand-off, call and result), a timeline per agent and a live log; then the brief: the
    answer, the findings it rests on with their figures and code's reruns, what the figures can't
    tell, a follow-up box and the link. Plain HTML and JavaScript, no build step: the stream is
    read with `fetch` (a POST, so not `EventSource`). Mocked first with a replayed example run and
    agreed with the user before it was wired up.
  - The question runs in a task of its own, its record saved at the start, after each finding
    and at the end. **Closing the page stops it** (decided 2026-11-13): a serverless host keeps a
    machine only while requests are open, so a question left running after its stream closed
    would be cut off. What it found is saved, status "stopped". A comment line every
    15 s keeps the stream open and notices a closed page.
  - `GET /briefs/{id}` (by link; "running" with the findings so far while it runs), `/examples`,
    `/profiles`, `/profiles/{name}`, `/health`: no key, no tokens.
  - Refused before anything runs: a key Groq refuses or that can't use the models (401, checked
    with Groq's model list, no tokens); a second question on the same key while one runs, on any
    machine (409: a lease in the store); more than `STATS_DESK_MAX_RUNNING` (10, to measure)
    questions on one machine (503, Retry-After).
  - **Free questions** (decided 2026-11-13, user): with no key, a question runs on one of the
    desk's own keys (`STATS_DESK_FREE_GROQ_KEYS`, each on a different Groq organisation, since
    limits are per organisation), one not busy with another question.
    - Per visitor: `STATS_DESK_FREE_PER_VISITOR_PER_DAY` (3, new questions and follow-ups
      together), by a hash of their address, never the address, kept in the store so every
      machine agrees. No fixed number of questions per key.
    - Which key: the free keys are used by the desk alone (user, 2026-11-13), so the desk counts
      each one's tokens over the last 24 hours itself (every model call adds to an hourly slot in
      the store; 25 slots summed: up to an hour too much, never too little). A question starts on
      the key with the most left, never on one under `STATS_DESK_FREE_MIN_LEFT` (20,000). Groq's
      replies can't tell us: they show the minute's remainder only. `/health` shows each free
      key's tokens left.
    - Calls Groq rejects (malformed tool calls, a few a question with gpt-oss) are charged but
      come with no token count (*measured* 2026-11-13: the error body holds message, code and
      the rejected text only; two rejected calls cost 447 and 567 of the minute's tokens with
      max_tokens 300 and 1,000, so what was written is charged, not the allowance). They go on
      the count as an estimate: that call's full context + the larger of the visible rejected
      text (counted exactly) and a typical reply for the agent (analyst 200, lead 300: above 9
      in 10 saved replies, which were 117 and 192-264; rejected calls ran longer).
    - Backstop: a key that runs dry mid-question anyway (Groq's "daily limit") is marked spent
      until Groq's Retry-After (every machine skips it), and the question carries on on the other
      free key at once (`llm.QuestionKeys`: the first analyst to hit the limit switches, the
      others retry on the new key). Not yet seen live: that Groq's message says "per day" and
      sends Retry-After (1 hour assumed if not).
    - With no free key left the question **pauses**: saved, and the visitor is told; their
      choice to carry it on with their own key (`POST /briefs/{id}/resume`) or leave it.
    - Worst case for abuse: the free questions run out for the day; the keys are free-tier, so
      nothing is charged. A VPN gets round the per-visitor count.
  - **Carrying a question on** (`POST /briefs/{id}/resume`; paused, or its page closed; at most
    3 times): a fresh lead conversation built from the record -- its findings so far, the tasks
    that didn't finish (listed, to give again if needed); new tasks numbered after the old ones;
    the earlier run's lead messages kept under their own name. A free question paused because the
    desk's keys ran dry doesn't count against the visitor again; one stopped by closing the page
    does. A question that ran on the user's own key needs that key to carry on.
  - Several machines can run side by side: what they must agree on (key leases, free counts,
    records) is in the store; each machine has its own governor buckets, which is right because
    one key runs one question at a time.
  - A question streams for ~5-6 min: the host must not cut a request off by its length (see the
    host, below).
  - `scripts/publish_brief.py <id> [--off]` makes a brief a public example.
- **The analyses**: one shared analysis server per machine (2026-11-13), MCP over HTTP on the
  loopback, started with the app and restarted if it dies; each question opens its own
  connection and keeps its own results. Replaced one stdio child per question (the desktop-app
  pattern: a Python and DuckDB process per question). Scripts and evals still start their own.
- **Who sees what** (decided 2026-11-10): a few chosen briefs are public examples, so a visitor
  without a Groq key sees the product working; any other brief opens only by its link.
- **Web page**: to the user's details (pending).
- **Async**: analyst instances in parallel; MoSPI fetched one page at a time by the release job; DuckDB in a worker thread; one event loop for many users.
- **Guarding our own server**: a cap on questions running at once on each machine (in memory).
- **Docker** (from step 3, decided 2026-11-10; `Dockerfile` written 2026-11-12; built and run
  2026-11-13: 437 MB, non-root user `desk`, no `.env` inside, both servers up in ~7 s, health
  check healthy, the pages and the 401 for a keyless question as expected; the first build found
  a bug -- `COPY --chmod=444` gave the folders it created 444 too, so the app couldn't reach the
  release file -- fixed by creating them first). Locally: `docker run -p 18080:8080 stats-desk`
  (port 8080 is taken on the dev machine). The same image is tested and rebuilt by the release
  job and served by Azure Container Apps. Two stages (install with uv's lock file, then a slim
  runtime), non-root, release file read-only, health check, one worker. `.dockerignore` keeps
  every `.env` out of the image.
- **Store** (`api/store.py`): Azure Cosmos DB when `STATS_DESK_STORE=cosmos`, in memory otherwise
  (tests, local runs). Question records (the whole record as one JSON field, so the database
  indexes a handful of fields, not every message), key leases, daily visitor counts, free keys'
  hourly token counts, spent-key blocks. Never a key or an address. Both stores pass the same
  checks (`tests/test_store_contract.py`; on Cosmos 2026-11-14, 7/7, including four claims of one
  key at once -- one wins -- and counts written at once adding up).
- Settings: `.env.example` lists every variable.
- **Host: Azure** (decided 2026-11-14). Google Cloud was the plan, but its billing sign-up
  refused the account with OR_BACR2_59 (Google's own check, after the card's ₹2 test charge
  went through; Google's forums show the same code across accounts and cards, no fix but its
  billing support). Azure's free account went through. Everything in resource group
  `stats-desk`, Central India (Pune):
  - **Azure Container Apps** (Cloud Run's counterpart): scales to zero; free each month 180,000
    vCPU-seconds, 360,000 GiB-seconds, 2 million requests. Its front end ends a request after 240 s
    of SILENCE: the stream's comment every 15 s should keep it open -- to confirm on the first
    deploy (only a third party's lab says so).
  - **Azure Cosmos DB, free tier** (one account per subscription, chosen at creation, for the
    life of the account): account `statsdesk60c821`, database `stats_desk` at a FIXED 1,000 RU/s
    (the free amount; never autoscale), shared by containers `briefs`, `leases`, `counters`,
    `blocks`, `usage` (partitioned by `id`). Signed in to by identity (no password): the app's
    managed identity on Azure, `az login` locally, each with the "Cosmos DB Built-in Data
    Contributor" role on the account.
  - **The image on GitHub Container Registry**, not Azure Container Registry (which has no free
    tier, ~₹400 a month).
  - **Deployed 2026-11-14**: Container App `stats-desk` in environment `stats-desk-env` (no paid
    logs), https://stats-desk.kindrock-535a71d1.centralindia.azurecontainerapps.io, image
    `ghcr.io/venkatdatta2402/stats-desk:2026-11-14` (public). 0.5 CPU / 1 GB, 0 to 1 replica,
    scales to zero 300 s after the last request (user kept 5 min: people read and think between
    questions). Its system identity has the Cosmos data role; the free Groq keys and Langfuse
    keys are secrets on the app. Woke from zero and answered in 5 s; reads Cosmos by identity.
  - **What it can cost** (2026-11-14): only Container Apps, billed while awake. One question ≈ 6
    min answering + 5 min idle ≈ 330 vCPU-s and 660 GiB-s, so the free grant covers ~500 questions
    a month run alone (more when they overlap), or ~1,000 page wakes. Cosmos can't cost (fixed at
    the free 1,000 RU/s; over it, requests wait, not bill); egress (100 GB free), identity, image,
    budget: free. Worst case, something keeping it awake all month: ~₹2,500-3,000 (third-party
    per-second prices, approximate).
  - **At once** (current settings): up to 10 questions on the one machine (`STATS_DESK_MAX_RUNNING`),
    at most 2 of them free (one per free key).
  - **Long streams** (*measured* 2026-11-14, a temporary test app in the same environment, then
    deleted): a silent stream was cut at 241 s; one with a comment every 15 s ran its full 330 s.
    So the cutoff is real, and the app's comments are what let a long question finish.
  - **How many fit** (*measured* 2026-11-14): the image in Docker limited to 0.5 CPU / 1 GB as on
    Azure, against a stand-in for Groq (2-6 s a reply, the real agents, analysis server and
    checks; `STATS_DESK_GROQ_BASE_URL` points the app at it): every question finished at 1, 2, 5,
    10, 15, 20 and 30 at once, in ~80 s each (as fast as one alone). Memory 245 MB idle, 272 MB
    at 10, 352 MB at 30; CPU reached the 0.5 limit from 15 on without slowing anything (the
    questions mostly wait on the model). Live, with Langfuse and Cosmos, one question peaked at
    391 MB, so the real figures run higher. The limit is the keys, not the machine: one question
    per key, two free keys.
  - **First live question** (2026-11-14, free, "Is unemployment among young people rising?"):
    brief in 120 s, 22,609 tokens, 9 model calls, 2 tasks; the stream's 15 s comments arrived
    on time; record saved in Cosmos; the free key's own count went down by exactly 22,609; the
    visitor count keyed by the caller's own address (X-Forwarded-For's last entry is right on
    Container Apps). Azure's metrics: CPU peak 0.048 of 0.5 cores, memory peak 391 MB of 1 GB
    (app + shared analysis server, one question). Quality: the brief set the annual figure (usual status, 2024 to 2025, fell
    0.4) against the monthly one (weekly status, Apr 2025 to Sep 2026, rising) as a
    contradiction, without saying they are different measures and periods.
  - **Cost guard** (user, 2026-11-14): the hard limits above, and a **₹1 monthly budget** on the
    subscription (`stats-desk-budget`): at 100%, one email to the owner and the action group
    `stop-stats-desk`, which runs Logic App `stop-stats-desk` (its identity is Contributor on the
    app only) calling Container Apps' stop action. Tested by firing the Logic App by hand: the
    app stopped, the site went 404; started again with the `start` action. Free grants never show
    as cost, so ₹1 means "the first rupee of real charges". Azure's cost figures lag 8-24 h: a
    few rupees of overrun is possible.
  - The free account's ₹19,045 credit ends 14 Dec 2026: it must be upgraded to pay-as-you-go
    before then, or Azure switches the subscription off (the always-free amounts apply only after
    upgrading).
- **Storage, all free**: release tables and profiles inside the Docker image (rebuilt by the
  release job); Cosmos DB free tier for records, leases and counts.
- **No Redis** (decided 2026-11-10). Every job it could do here is done already: follow-ups and
  saved briefs by the store (read once when a follow-up starts, written as a question goes:
  Redis's speed saves milliseconds on a question that takes minutes, and briefs must be kept
  permanently anyway); limits by Groq on each user's key; the per-minute bucket in memory, right
  because one key runs one question at a time.

---

## 9. Safety and honesty rules (enforced in code)

1. No number without a published cell behind it.
2. Never mix measures (CWS / PS+SS), base years or overlapping age bands.
3. Never average or add rates; MoSPI's total rows are the totals.
4. Only fields with a total can be dropped from a split.
5. Gaps are said, never filled.
6. Correlations describe states, not causes; built-in ones are not findings.
7. Small or outlying groups are flagged, never headlined.
8. A release with a new shape stops the profile build.

---

## 10. Testing and evaluation

- **Offline, no tokens** (63 tests): governor, agent loop and tools (on a scripted model), the
  MoSPI client (paging, errors returned as success, rate limits, breaker), profiles and the
  release file (real 2025/2024 PLFS numbers in a trimmed fixture), the guard, every analysis,
  the server over MCP (in-process, and as a child process on the real release file), the
  hand-in number check.
- **Live**: every run saved before anything judges it.
- **Observability** (built 2026-11-11, `core/trace.py`, from ask-the-cookbook's traces.py): one
  record per question of every model call (agent, tokens, seconds, seconds waiting for Groq's
  minute limit, Groq's error) and every analysis run (by an agent or by code, arguments, result
  id or refusal, seconds), how it ended, totals and cost if paid. Kept with the run, a log line
  without the question's text, and Langfuse (this project's own; on only when its keys are set;
  never able to break a run). In Langfuse each question is a tree: the lead, its ask_analysts
  call, under it each analyst and its calls.
- **Evaluation** (built 2026-11-11, `evals/`; no model judge: the numbers are checkable by code).
  8 questions (`evals/data/golden.json`): a gap, a trend, a comparison across states, a
  combination MoSPI doesn't publish, and 4 traps (a link true by definition, a pattern that held
  one year only, a top figure from a small-sample outlier, a question asking for a cause). Every
  answer-key number comes from an analysis call, and a test reruns them on the release file;
  another test keeps the questions and their answers out of every prompt and tool description.
  Q1 is the question the agents were tuned on (live runs 1-3): not a clean test.
  - outcome, by code: a brief or not; answer-key points made (their numbers in the brief); the
    measure named; the traps' checks; numbers in no result; numbers only in a profile; cause words
  - path, by code: reference steps covered by the first result holding their numbers; precision
    over the analyses the agents chose; wasted analyses; findings built from chained single
    figures; the lead's tasks listed for marking by hand (hypothesis or lookup)
  - cost, from the trace: tokens by agent, calls, seconds, waiting, Groq's errors, tasks stopped
    or refused
  - `python -m evals.collect_runs` (spends ~40-60k a question; resumable across days) then
    `python -m evals.score evals/runs/<file>.jsonl` (free) -> `evals/reports/<file>.md`
  - Not measured: reliability (one run per question on the free tier).

---

## 11. Layout

Code is grouped by the job it does; everything the code produces lives under `data/`.

```
projects/stats_desk/
├── pyproject.toml, uv.lock          own dependencies (uv)
├── DESIGN.md                        this file
├── src/stats_desk/
│   ├── config.py
│   ├── core/       used by the agents: AI calls (llm), governor (Groq limits), meter,
│   │               events, tools, agent loop (runtime)
│   ├── release/    the release job: MoSPI client (mospi), lookups and totals (cells),
│   │               statistics (stats), profiles (build), the release file (store)
│   ├── analysis/   the analysis MCP server: reading requests (data), the calculations
│   │               shared with the profiles (compute), the tools' answers (analyses),
│   │               the server, the agents' side of it (client)
│   ├── agents/     Agent, forms, answer (agent); analyst; code's checks (checks); the
│   │               question's record (record); lead, investigation, follow-ups (lead)
│   └── api/        the web layer (app); questions: starting, running, keys, carrying on
│                   (desk); the machine's analysis server (analysis_server); records, key
│                   leases, visitor counts, key usage (store); progress lines (progress)
├── scripts/        fetch_release_data, build_release, ask (one live question)
├── runs/           saved live runs: brief, transcripts, results, usage
├── data/
│   ├── raw/        downloads from MoSPI (not committed)
│   ├── release/    plfs.duckdb: the stored tables, their cards, the profiles
│   └── profiles/   the profiles as text, for reading and diffing in git
└── tests/          offline tests; a trimmed copy of real PLFS as fixture
```

## 12. Status and build order

| Done | |
|---|---|
| Async core: model client, governor, meter, runtime, tools | ✓ |
| MoSPI client for the release job (paging, pacing, errors, rate limits) | ✓ |
| Profile engine for PLFS (UR, LFPR, WPR; annual and monthly) | ✓ built from live data |
| Release file `data/release/plfs.duckdb`: 6 cleaned tables, a card each, the profiles (2.6 MB, built in seconds) | ✓ |
| Analysis MCP server: 9 analyses, refusals that say what's published, results by id, hand-in number check | ✓ |
| Profiles built from the stored tables with the tools' own calculations (one copy of each rule) | ✓ |
| Offline tests for all of the above | ✓ 63 pass |

| To build, in order | Tokens |
|---|---|
| 1. ~~Analysis MCP server~~ done 2026-11-10 | 0 |
| 2. Lead, general analyst (parallel instances), skeptic; the brief | live tests ~50-100k |
| 3. ~~API with SSE, user's Groq key, governor per key; brief store; Dockerfile~~ written 2026-11-12; ~~one record per question, follow-ups from it, free questions on the desk's keys, shared analysis server, several machines~~ 2026-11-13; image built and run 2026-11-14; host moved to Azure, Cosmos DB store passing its checks live (132 tests + 7 live) | 0 |
| 4. Azure Container Apps deploy (image on GitHub Container Registry, managed identity, cost guard), release job on GitHub Actions | 0 |
| 5. Evaluation set and runs | ~200k, spread over days |
| 6. History back to 2017-18; more datasets (CPI first); web page | — |

Removed 2026-11-09 (archived outside the repo, in _archive/stats_desk_day1): the day-1 subject
analysts and their MoSPI tools, notebooks, "fetches that worked", free SQL and its hand-in form,
the per-question DuckDB sandbox, the day-1 live-run script, recorded responses and runs.

## 13. Open decisions

1. Web page details.
2. A free "what's new" page straight from the profiles, with no AI?
3. Which datasets after PLFS (CPI suggested first, for real wages).
