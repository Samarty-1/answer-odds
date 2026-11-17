# Protocol

Written and committed before any model was scored on the test period. The git history
shows this file landing before `results/`.

## Question

When a question is posted on Ask Ubuntu, how well can the chance that it gets an
answer within 7 days be predicted, using only what is known at the moment it is posted?

## Data

Stack Exchange data dump for askubuntu.com, published April 2024 (last post
2024-03-31). Unit: one question. Outcome `answered_7d`: an answer still present in the
dump was posted within 7 x 24 hours of the question. Self-answers count.

Left out:

- questions whose asker is missing from the dump (7,516 of 414,451). The dump blanks
  the owner when an account is deleted later, so "no owner" is information from the
  future;
- questions asked in the last 7 days of the dump, whose outcome is not yet settled;
- questions asked before 2011-01-01 (the site's first months; kept as history for the
  features, not scored).

## Split, by time asked

| Part | Period | Used for |
|---|---|---|
| Train | 2011-01-01 to 2020-12-31 | fitting |
| Validation | 2021-01-01 to 2021-12-31 | choosing settings, early stopping, calibration check |
| Test | 2022-01-01 to 2024-03-24 | scored once, at the end |

No shuffling. Nothing about the test period is inspected before the final run: not its
base rate, not its feature distributions.

## Features

History features come from one SQL function, `feat.question_features(user, tags, time)`,
which reads running totals "as of the start of the day the question was asked". A
question's own 7-day outcome only enters those totals 7 days after it was asked.
Text features are simple counts computed from the title and body (lengths, code
blocks, links, images). No feature uses any column that is updated after posting.

## Models

- **B0** constant: the training base rate.
- **B1** site rate: share of questions answered, outcomes settled in the previous 30 days.
- **B2** tag rate: mean 365-day answer rate of the question's tags (one feature).
- **M1** logistic regression on all features (standardised, missing values imputed with
  the training median plus a missing flag).
- **M2** LightGBM on all features.
- **M3** logistic regression on TF-IDF of title + body only (no history), to see what
  text alone carries.

Settings for M1 to M3 are chosen on validation from a small fixed grid (at most 12
configurations per model, all recorded in `results/`). The chosen model is refit on
train + validation before the test run.

## Metrics

Primary: ROC AUC on test. Secondary: Brier score, log loss, and calibration (predicted
vs observed rate in ten equal-count bins). Intervals: 1,000 bootstrap resamples of the
test questions.

## Claims fixed in advance

1. **M2 is useful** if its test AUC is at least 0.65 and it beats B2 by at least 0.02
   AUC, with the bootstrap 95% interval of the difference above zero. Otherwise the
   write-up says the history and text features add little beyond the tag.
2. **Calibration holds** if M2's test Brier score is lower than B1's (the rolling site
   rate, which tracks drift in the base rate by construction). If not, the write-up
   says the probabilities are not trustworthy out of period, whatever the AUC.

## Leakage audit

Three ways to leak are added to M2 one at a time, same settings, same split, and the
test AUC of each is reported next to the honest model:

- **A, asker snapshot**: `users.reputation`, `views`, `up_votes`, `down_votes` as they
  stood in April 2024, not when the question was asked.
- **B, future-aware history**: the same rate features computed over the whole dataset
  (including later questions, and the question itself) instead of as-of the asking day.
- **C, post snapshot**: the question's `view_count`, `score`, `comment_count` and
  `favorite_count`, all of which accumulate after posting.

These models are not candidates. They exist to put a number on how much each shortcut
would have inflated the result.

## Known limits, stated now

- Deleted questions are gone from the dump: 37% of all post ids up to the maximum are
  missing. Deleted questions are more likely to have gone unanswered, so every rate
  here is higher than what a live moderator would see.
- Answers that were later deleted do not count as answers.
- A closed question (for example a duplicate) rarely gets an answer. Closing happens
  after posting and is not a feature, but part of what the model learns is "this looks
  like a question that gets closed".
- One site. Nothing here says the result carries to another community.
