# Where to spend an LLM on training data: generating, labelling and checking for drug-drug interaction extraction

Gwilym Hughes, AI4BioMed Lab, University of Glasgow. EPSRC Vacation Internship, summer 2026, supervised by Jake Lever.

A large language model can supply training data for a smaller classifier in two ways: it can write new labelled sentences, or it can label sentences that already exist. We compared the two for drug-drug interaction extraction on DDI-2013, and added a third step in which the model checks its own positive labels.

On the held-out test set, a BiomedBERT classifier trained on sentences our best generator wrote scored 0.528 micro-F1, 65% of the 0.806 reached with the human-annotated training set. Trained on the same model's labels for the real training sentences, it scored 0.680, and once the model had checked its positive labels, 0.728, 90% of full human supervision, with no human annotation at all. With human labels on 1,000 of the 2,402 training sentences and checked model labels on the rest, the score reached 0.773, 96%. The sections below give the method and evidence for each figure, and the accompanying tutorial shows the sentences and pairs behind the two central results.

<!-- Introduction: deferred to the paper-targeting step. It should cover the annotation bottleneck, the distillation premise, Kazemi et al. (2025) as the nearest prior comparison, and what a pair-level task adds. -->

## 1. Task and evaluation

**Task.** Each sentence arrives with its drug mentions marked, and every unordered pair of mentions receives one of five labels: MECHANISM, EFFECT, ADVISE, INT or NONE. The label belongs to the pair, so a sentence naming four drugs yields six instances. About five pairs in six are NONE, so scores are micro-averaged $F_1$ over the four positive labels.

**Data.** The DDI-2013 training directory is split at document level into training (70%), development (15%) and validation (15%) sets. The official test directory is the test set.

| set | pairs | used for |
|---|---|---|
| development | 4,243 | designing and comparing generators |
| validation | 6,276 | choosing between configurations |
| test | 6,744 (976 positive) | one evaluation, of the configurations validation chose |

The choice was written to disk and committed before the test run. We do not apply the negative instance filtering common in published work on this corpus, which deletes trivially negative pairs before training and evaluation. Published scores on the benchmark range from about 65 to 90, much of that spread reflecting preprocessing, and our scores are not comparable with them.

**Classifier.** Every configuration fine-tunes the same model, `BiomedBERT-base-uncased-abstract-fulltext`, with learning rate $2\times10^{-5}$, batch size 32 and maximum length 256. It trains for three epochs or 500 optimisation steps, whichever is more; section 6 explains why the floor matters. We report the mean over three to five seeds. A difference between two configurations is given with its number of standard errors, where

$$\text{SE}(\bar{a} - \bar{b}) = \sqrt{\frac{s_a^2}{n_a} + \frac{s_b^2}{n_b}}.$$

Each score carries the set it was measured on: [test], [val] or [dev]. No comparison crosses sets.

## 2. Writing labelled sentences

**Method.** A generated sentence starts as a specification: the drugs it must mention, drawn from 14,932 DrugBank names and 425 WHO-ATC groups, and which pairs among them stand in which relation. The specification is rendered as a prompt and sent to `gpt-oss-120b` (temperature 0.9, low reasoning effort). A deterministic second stage finds each drug in the output, tags every pair, and copies its label from the specification. A sentence missing any required drug is rejected. The specification is written before the sentence, so it is the gold annotation, and labels are correct by construction. *Tutorial, section 2.*

**Versions.** Five versions of the generator were trained and scored [dev].

| version | design | $F_1$ | $P$ | $R$ | seeds |
|---|---|---|---|---|---|
| v13 | one sentence expressing a given relation between two given drugs | 0.279 | 0.184 | 0.588 | 3 |
| v14 | adds drugs outside the relation, so sentences contain NONE pairs | 0.359 | 0.288 | 0.476 | 3 |
| v15 | draws sentence content to match the corpus | 0.333 | 0.272 | 0.430 | 3 |
| v17 | builds each sentence from one of nineteen frames derived from the annotation guidelines | 0.391 | 0.381 | 0.403 | 3 |
| v18 | separates label-carrying assertions from label-neutral modifiers | 0.486 | 0.398 | 0.628 | 5 |

![Generator trajectory](figures/fig2_trajectory_logged.png)

*Figure 2. Development scores of each generator version. The dashed line is the same classifier trained on the human-annotated training set, about 0.80 [dev].*

**The pair-count shortcut.** In v13, sentences containing one pair were 74.8% positive and sentences containing two or three pairs 0.2% positive, so the number of drugs in a sentence predicted its label. v14 added drugs that take no part in the relation and removed the shortcut. Precision rose from 0.184 to 0.288, and recall fell. *Tutorial, section 2.2.*

**The construction shortcut.** In v17 every frame produced the same sentence label every time. With $Y$ the sentence label and $F$ the frame, the share of the label's uncertainty that the frame removes is the uncertainty coefficient

$$U(Y \mid F) = \frac{H(Y) - H(Y \mid F)}{H(Y)},$$

which is 1 when the frame determines the label. For v17 it is 1.000. The frame was also visible in the finished text: a bag-of-words model recovered it from sentences with drug names masked at 0.961 accuracy over 19 frames, so a classifier could read the label off a sentence's wording without assessing whether the two drugs interact. For 300 corpus sentences classified by construction (by `gpt-oss-120b`), the same coefficient is 0.464. Real constructions carry some information about the label, and that level, not zero, is the target.

v18 draws one or two of seventeen assertions per sentence, each binding a pair and carrying a label, and independently draws from eight modifiers that carry none. A variant applied to 26% of positive assertions retargets them at a drug outside the sentence's list, which makes the listed pair NONE. v18's coefficient is 0.458. Its $F_1$ rose to 0.486, the gain lying in recall (0.403 to 0.628) with precision nearly unchanged. The coefficient differs from symmetric normalised mutual information, which divides by the mean of both entropies and gives 0.398 on the same v17 specifications. *Tutorial, section 2.3, which builds the measure from a single v17 sentence.*

**Other results.** v15 moved closer to the corpus on every distributional measure we took, and its $F_1$ fell. Generating v18 at high reasoning effort scored 0.453 against 0.462 at low effort at matched size, 0.5 standard errors apart, at several times the cost per request [dev]. On the test set v18 scored 0.528 [test].

## 3. Labelling real sentences

**Method.** The comparison arm keeps the real training sentences and their gold entity spans, discards the gold labels, and asks the same model to label every pair. The prompt gives the five labels and the guideline conventions that decide difficult cases, such as that a pair is NONE unless both drugs take part in the asserted interaction. Requests carry eight pairs each, at temperature 0 and high reasoning effort. Whole-sentence requests had lost 26% of pairs, disproportionately from dense sentences. 2,402 of 2,417 eligible sentences were labelled completely. Against gold, the model's labels on these sentences have precision 0.623 and recall 0.881. *Tutorial, section 3.*

**Label cost against text cost.** On the test set, a classifier trained on the 2,402 sentences with human labels scored 0.806, and with the model's labels on the same sentences, spans and pairs, 0.680 [test]. This is the only exactly matched comparison in the project, and it puts the cost of the labels at 0.126 (23.7 standard errors). Trained on generated text instead, the classifier scored 0.528, a further 0.152 lower, although the generated labels are correct by construction and the generated set has 2.3 times as many training pairs [test]. An earlier development comparison with 2,248 sentences per arm, matched in count but drawn independently, gives the same ordering: 0.781, 0.661 and 0.444 [dev].

![Label and text provenance](figures/fig1_provenance_test.png)

*Figure 1. Test scores by where the training labels and text came from. The first three bars share the same 2,402 real sentences.*

## 4. Checking the labels

**Method.** The model's labelling errors are mostly false positives, so only its positive labels are checked. A second call receives the sentence with every drug mention numbered inline and one question per pair: would the guidelines annotate an interaction between these two mentions? Mentions are numbered rather than named because a drug named twice is two mentions, of which the guidelines let only one take part. A rejected pair is demoted to NONE. At high reasoning effort the verifier recalled 0.920 of true NONE pairs and 0.948 of true positives on human-labelled development data. During its development, numbering mentions inline and adding the guideline rules to the prompt raised NONE recall from 0.532 to 0.902 with same-entity pairs excluded; the two changes were made together. *Tutorial, section 4, which shows the exact question for one sentence.*

**On the model's labels.** The verifier rejected 614 of the model's 3,573 positive labels, 17.2%. 174 requests failed, and the positives in those sentences stayed unchecked. On the test set, checking moved precision from 0.578 to 0.676 and recall from 0.824 to 0.790, and $F_1$ from 0.680 to 0.728 (9.5 standard errors), recovering 39% of the label cost [test]. On validation, a control that lowered the positive rate at random gained 0.013 (1.1 standard errors) where targeted demotion gained 0.094, so the gain depends on which positives are removed, not how many [val]. The gain from checking was 0.079 on validation and 0.049 on test, a shrinkage to expect after selecting on validation.

**On generated text.** Removing the pairs the verifier flagged beat removing the same number at random by 0.041 for v14 and 0.042 for v15 [dev], but by 0.011 for v18 (1.1 standard errors, five seeds) [val]. All 878 pairs removed from v18 were NONE by construction, and they came disproportionately from the variant that writes positive-sounding NONE sentences on purpose: 4.8% of those sentences' pairs were removed against 1.9% for sentences without variants. On v18 the verifier removed deliberate hard negatives rather than errors.

## 5. Spending annotation

**Validation grid.** For human budgets of 0 to 1,000 sentences, the grid compared adding nothing, the model's labels on the remaining sentences, checked model labels, v18 text, or model labels and v18 together [val]. Checked model labels were the best addition at every budget. v18 added 0.072 to 100 human sentences (2.8 standard errors) and nothing from 250 upwards, and adding it to model labels lowered $F_1$ at every budget, by 0.015 to 0.026.

![Mixing grid](figures/fig3_grid.png)

*Figure 3. The mixing grid on validation. Bars are one standard deviation across seeds.*

**Test.**

| training data | human sentences | $F_1$ | sd | $P$ | $R$ | share of all human |
|---|---|---|---|---|---|---|
| generated v18 | 0 | 0.528 | 0.021 | 0.465 | 0.611 | 65.5% |
| model labels | 0 | 0.680 | 0.011 | 0.578 | 0.824 | 84.4% |
| model labels, checked | 0 | 0.728 | 0.005 | 0.676 | 0.790 | 90.4% |
| human labels | 1,000 | 0.745 | 0.026 | 0.773 | 0.720 | 92.5% |
| human labels + checked model labels | 1,000 | 0.773 | 0.010 | 0.742 | 0.807 | 96.0% |
| human labels | 2,402 | 0.806 | 0.006 | 0.797 | 0.815 | 100% |

*Table 3. Five seeds each, on the official test set.*

![Annotation cost against test F1](figures/fig4_test_cost.png)

*Figure 4. Test $F_1$ against the number of human-labelled training sentences.*

With no human annotation, checked model labels came within 0.02 of 1,000 human-labelled sentences (−0.017, 1.4 standard errors). With 1,000 human sentences, 42% of the training set, they added 0.028 (2.2 standard errors) and fell 0.032 short of full annotation (6.2 standard errors). The 1,000-sentence human arm varied two to five times more across seeds than the checked arms, because each seed drew different sentences. The ordering validation chose held on test at both budgets.

v18 scored 0.063 higher on test than on validation. The test set is 88% DrugBank text, the register v18 imitates. On development, v17 scored 0.396 on DrugBank sentences and 0.144 on MedLine.

## 6. How the approach developed, and corrections

Each version and experiment tested a working assumption, as follows.

| dates | assumption | test | outcome |
|---|---|---|---|
| 23 to 27 July | class balance limits generated data | NONE-ratio sweep | flat, 0.25 to 0.28 [dev] |
| 27 July | generated data loses through quantity or balance | human data cut to matched size and ratio | 0.460 at 300 positives [dev]; the gap is the text |
| early August | pair count predicts the label | positive rate by pairs per sentence | confirmed; v14 |
| 14 to 15 August | generated data is worth adding to small human budgets | mixing curve | an apparent +0.52 at 100 sentences; corrected below |
| 16 August | matching the corpus's distribution raises $F_1$ | v15 | every distance closed, $F_1$ fell |
| 17 to 18 August | label fidelity explains the gap | verifier pruning with a random control | +0.04 on v14 and v15 [dev] |
| 18 to 25 August | missing constructions explain the gap | v17, then v18 | construction shortcut found and removed |
| 30 August | labelling real text beats generating it | the comparison in section 3 | confirmed |
| September | checking the model's labels closes most of the gap | grid, then test | confirmed; generated text does not complement |

The comparison in section 3 was run after five generator versions; it should have been run first.

**Corrections to earlier figures.**

| earlier claim | correction |
|---|---|
| generated data worth +0.52 at 100 human sentences [dev] | the human-only arm was trained for a fixed three epochs, about 60 optimisation steps, and predicted NONE for every pair in four of five seeds. With the step floor, v18 adds +0.072 [val] |
| model labels beat human labels at 250 sentences [dev] | both arms had about 160 steps; not established |
| v14 at 0.379, v17 at 0.376 | no run records exist; the logged figures are 0.359 and 0.391 |
| frame and label "NMI 1.000" | this is the uncertainty coefficient; symmetric NMI on the same data is 0.398 |

## 7. Limitations

- One corpus, one generating and labelling model, one classifier.
- Gold entity spans throughout, so every configuration assumes a named entity recogniser.
- No negative instance filtering.
- Three to five seeds per configuration. Reruns of the same design differed by up to 0.02.
- Development was reused across the generator work, and gains measured on validation shrank on test.
- The corpus construction labels behind the 0.464 target were assigned by the model.
- The verifier was calibrated on development data, and 174 of its requests failed.
- The test set is 88% DrugBank.
- No training runs were logged between 31 July and 13 August, so results from that period rest on written notes.

## 8. Open questions

- What fraction of the 614 demoted pairs are NONE in the gold labels? The tutorial computes it on a 400-sentence sample; the full count is not yet measured.
- How much of the verifier's gain in NONE recall comes from the guideline rules and how much from numbering mentions?
- 21% of gold-negative sentences carry constructions that usually indicate an interaction, per the model's construction labels. How many are annotation errors?
- Does combining generators (0.530 for v14 and v18 against 0.484 for v18 alone at matched size [dev]) survive on test?

<!-- Discussion: deferred to the paper-targeting step. -->

## Resources

- **Tutorial:** `tutorial/ddi_tutorial.ipynb`, runnable on a laptop with the committed samples. Section 2.3 builds the construction shortcut from one sentence; section 4 shows the verifier's question and verdicts for one sentence.
- **Code and run records:** the repository, including every training run with its configuration, metrics and code state.
- **Full ledger:** `ddi_results.html`, all 470 runs grouped into experiments, with provenance.

## References

Herrero-Zazo, M., Segura-Bedmar, I., Martínez, P. and Declerck, T. (2013). The DDI corpus: an annotated corpus with pharmacological substances and drug-drug interactions. *Journal of Biomedical Informatics*, 46(5), 914-920.

Kazemi, A. et al. (2025). Synthetic vs. Gold: the role of LLM-generated labels and data in cyberbullying detection. arXiv:2502.15860.
