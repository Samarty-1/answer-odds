# answer-odds

Will a new Ask Ubuntu question get an answer within a week? A model that predicts it
from what is known at the moment of posting, a point-in-time feature store in
PostgreSQL that guarantees that "known at the moment", and an API that serves the
model through the same SQL that built its training data.

The evaluation was fixed before the test period was touched
([docs/PROTOCOL.md](docs/PROTOCOL.md), committed first). **The model fails the bar
the protocol set for it.** The result below is what it actually does.

## Result

Test period: questions asked 2022-01-01 to 2024-03-24 (40,133), scored once by a model
fitted on 2011 to 2021 (361,533). Intervals: 1,000 bootstrap resamples.

| Model | AUC | 95% interval | Brier | Mean predicted |
|---|---:|---:|---:|---:|
| B0 constant (training base rate) | 0.500 | - | 0.304 | 0.655 |
| B1 site answer rate, previous 30 days | 0.573 | 0.567 to 0.579 | 0.237 | 0.409 |
| B2 answer rate of the question's tags, previous 365 days | 0.625 | 0.620 to 0.631 | 0.232 | 0.442 |
| M1 logistic regression, all features | 0.635 | 0.629 to 0.640 | 0.228 | 0.416 |
| **M2 LightGBM, all features** | **0.639** | 0.633 to 0.644 | **0.227** | 0.445 |
| M3 TF-IDF of title and body only | 0.626 | 0.620 to 0.632 | 0.263 | 0.581 |

Observed answer rate in the test period: 0.402.

Claims fixed in advance:

1. **"M2 is useful"** needed a test AUC of at least 0.65 and a lead over B2 of at
   least 0.02. **Fails**: 0.639, and +0.013 over the tag rate (95% interval +0.009
   to +0.018). The lead is real but small.
2. **"Calibration holds"** needed a Brier score below B1's. **Holds**: 0.227
   against 0.237.

What that means in plain terms: one number, how often questions with these tags got
answered last year, does most of the work. Everything else (the asker's history,
length, code blocks, error messages, time of day) adds about 0.013 AUC on top. In the
trained model the tag rate alone carries more gain than the next nine features put
together.

**The site is drying up, and the model lags it.** The share of questions answered
within a week fell every year: 80% in 2011, 52% in 2020, 33% in 2024. M2's
predictions are 3 to 5 points too high in every decile of the test period, because it
learned from easier years. It still beats the rolling site rate on Brier, because it
ranks questions better, but anyone using these as probabilities should shift them down.

## Leakage audit

Three shortcuts, each added to M2 alone with the same settings and the same split
(full figures in `results/test.json`):

| Shortcut | What it uses | Test AUC |
|---|---|---:|
| none (M2 as above) | only what was known at posting time | 0.639 |
| A asker snapshot | the asker's reputation and votes as of April 2024 | 0.686 |
| C post snapshot | the question's own views, score, comments, favourites | 0.714 |
| B future-aware history | tag and asker answer rates over all questions, including this one | **0.981** |

Shortcut B is the easiest to write by accident: a `GROUP BY user` over the whole
table. Three askers in four have exactly one question, so their "answer rate" *is* the label.
It would have turned a 0.64 result into a 0.98 one.

## How "known at posting time" is enforced

`sql/020_daily.sql` stores running totals per day for the site, each tag and each
user. Every event is filed under the day it became knowable:

- a question asked on day S is counted on day S;
- its 7-day outcome is filed under S + 7, the first day on which it is settled;
- an answer or a vote is filed under the day it happened, never before the post it
  belongs to.

`feat.question_features(user, tags, time)` (`sql/030_features.sql`) reads the
latest row *before* the asking day. Training calls it once per historical question
(`odds/build.py`); the API calls it once per request (`odds/api.py`). There is one
implementation, so the two cannot drift apart.

The tests check this directly. For sampled questions, they delete every row from the
asking day onwards, rebuild the whole feature store from what remains, and require
the features to be unchanged. That test caught a bug while this was being built:
votes generated with a date before their post were counted too early. (The real dump
has none, 0 of 3.67 million, but the SQL now guarantees it for any input.)

## Running it

Needs Python 3.11+ and PostgreSQL 14+. Uses the same loader as
[pg-query-lab](https://github.com/Samarty-1/pg-query-lab).

```bash
pip install -r requirements.txt
export ODDS_DSN=postgresql://postgres@localhost:5544/stacklab

curl -LO https://archive.org/download/stackexchange/askubuntu.com.7z
python -m odds.load extract askubuntu.com.7z data/xml
python -m odds.load load data/xml       # ~100 s
python -m odds.build                    # feature store + training table, ~2.5 min
python -m odds.dataset                  # pull the modelling frame (+ text) to data/
python -m odds.validate                 # fit on 2011-2020, choose settings on 2021
python -m odds.final                    # refit, score 2022-2024 once; refuses to run twice

uvicorn odds.api:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"title":"Grub rescue after Windows update","body":"<p>...</p>","tags":["grub2","dual-boot"],"user_id":15811}'

pytest -q    # 12 tests on a seeded made-up site; needs ODDS_TEST_DSN or the default local server
```

## Limits

- Deleted questions are not in the dump (37% of post ids up to the highest are
  missing), and deleted questions are likelier to have gone unanswered. Every rate
  here is higher than what a moderator would see live.
- Answers that were later deleted do not count. 464 answers predate their question,
  which is what merging questions produces; they count as answers.
- Questions whose asker's account was later deleted are excluded (7,516), because
  "no owner" is information from the future.
- Closed questions rarely get answers. Closure happens after posting and is not a
  feature, but part of what the model learns is what closable questions look like.
- Settings were chosen on 2021 alone. The grid was small (4, 8 and 3
  configurations) and every one tried is in `results/validation.json`.
- One site, one time split.

Data: Stack Exchange Data Dump (April 2024), CC BY-SA 4.0. No dump data is committed.
