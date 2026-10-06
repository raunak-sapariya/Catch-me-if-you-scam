# Email Spam Filter with Naive Bayes

A spam filter written from scratch in Python for a Probability and Statistics project. It uses **Bayes' theorem** and the **Naive Bayes classifier** to decide whether an email is *spam* (junk) or *ham* (a legitimate email). It is trained and tested on the public **Enron-Spam** dataset.

- **Accuracy: 98.94%** on 5,544 emails it never saw during training
- One file ([spam_filter.py](spam_filter.py))
- Every probability is computed by hand from word counts, so each formula below can be traced to a line of code

## Contents

1. [Quick start](#1-quick-start)
2. [Project structure](#2-project-structure)
3. [The dataset](#3-the-dataset)
4. [The theory: Naive Bayes step by step](#4-the-theory-naive-bayes-step-by-step)
5. [Worked example](#5-worked-example)
6. [How the code works](#6-how-the-code-works)
7. [Evaluation method](#7-evaluation-method)
8. [Results](#8-results)
9. [Limitations and possible improvements](#9-limitations-and-possible-improvements)
10. [References](#10-references)

---

## 1. Quick start

**Requirements:** Python 3.6 or newer (tested on Python 3.12.1). No packages need to be installed.

From the project folder, run:

```
python spam_filter.py
```

It takes about 5 seconds. The program:

1. loads 27,716 emails from `data/preprocessed/`
2. randomly splits them into 80% for training and 20% for testing
3. learns the probabilities from the training emails
4. classifies every test email and prints the accuracy, precision, recall and confusion matrix
5. prints the words that most strongly signal spam and ham
6. lets you type your own messages to classify (press Enter on an empty line to quit)

Output:

```
Loaded 27716 emails: 12671 spam, 15045 ham
Training on 22172 emails, testing on 5544 emails

Prior probabilities: P(spam) = 0.458, P(ham) = 0.542
Vocabulary size: 109619 words

Accuracy : 98.94%
Precision: 98.69%  (of emails marked spam, how many really are spam)
Recall   : 98.96%  (of all spam emails, how many were caught)

Confusion matrix:
              predicted ham  predicted spam
actual ham             3004              33
actual spam              26            2481

Top spam words: VIAGRA, VOIP, PILLS, CIALIS, OOKING, WIIL, WYSAK, UR, PHOTOSHOP, NBSP
Top ham words : HPL, ENRONXGATE, FASTOW, MMBTU, ENA, EES, ECT, DYNEGY, KAMINSKI, ENRON

Type a message to classify (or press Enter to quit): free money
-> SPAM  (P(spam | message) = 95.26%)
```

Because the random split uses a fixed seed (`random.seed(42)`), you should get exactly these numbers every time you run it.

---

## 2. Project structure

```
Spam/
├── spam_filter.py            the whole program
├── README.md                 this file
└── data/
    ├── readme.txt            the dataset authors' own description
    ├── preprocessed/         <- the program reads these
    │   ├── enron1/
    │   │   ├── ham/          3,672 legitimate emails, one .txt file each
    │   │   ├── spam/         1,500 spam emails, one .txt file each
    │   │   └── Summary.txt   who the emails belong to, date ranges, counts
    │   ├── enron2/           same layout
    │   ├── enron3/
    │   ├── enron5/
    │   └── enron6/
    ├── raw/                  the original, unprocessed emails (not used by the program)
    │   ├── ham/              beck-s, farmer-d, kaminski-v, kitchen-l, lokay-m, williams-w3
    │   └── spam/             BG, GP, SH
    └── archives/             the original .tar.gz downloads, kept as a backup
```

If the `data/preprocessed/` folder is missing, download the `enron*.tar.gz` files from the [dataset page](https://www2.aueb.gr/users/ion/data/enron-spam/) and extract them into `data/preprocessed/`, so that the paths look like `data/preprocessed/enron1/ham/0001.1999-12-10.farmer.ham.txt`.

---

## 3. The dataset

### Where it comes from

The **Enron-Spam** dataset was built by V. Metsis, I. Androutsopoulos and G. Paliouras for their paper *"Spam Filtering with Naive Bayes – Which Naive Bayes?"* (CEAS 2006).

- **Ham** comes from the mailboxes of six Enron employees. These emails became public during the investigation into the Enron company's collapse in 2001.
- **Spam** comes from three separate spam collections: **GP** (Georgios Paliouras), **BG** (Bruce Guenter) and **SH** (SpamAssassin + HoneyPot).

The authors combined these into six subsets, Enron1 to Enron6. Each subset pairs one employee's ham with spam from one of the collections.

### The subsets used here

| Subset | Ham (mailbox owner) | Spam source | Ham | Spam | Total |
|---|---|---|---:|---:|---:|
| enron1 | farmer-d | GP | 3,672 | 1,500 | 5,172 |
| enron2 | kaminski-v | SpamAssassin + HoneyPot | 4,361 | 1,496 | 5,857 |
| enron3 | kitchen-l | BG | 4,012 | 1,500 | 5,512 |
| enron5 | beck-s | SpamAssassin + HoneyPot | 1,500 | 3,675 | 5,175 |
| enron6 | lokay-m | BG | 1,500 | 4,500 | 6,000 |
| **Total** | | | **15,045** | **12,671** | **27,716** |

Enron1–3 have about 3 ham emails for every spam email, and Enron5–6 have about 3 spam emails for every ham email. Combined, the data is close to balanced: 54.3% ham, 45.7% spam.

### What a file looks like

Each email is a separate text file. The name `0001.1999-12-10.farmer.ham.txt` means: email number 0001 in order of arrival, sent on 1999-12-10, from the mailbox of farmer, and it is ham.

The authors have already cleaned the text. Each file starts with the `Subject:` line followed by the body, everything is lowercase, and punctuation is separated by spaces. Here is the start of a spam email:

```
Subject: dobmeos with hgh my energy level has gone up ! stukm
introducing
doctor - formulated
hgh
human growth hormone - also called hgh
is referred to in medical science as the master hormone . it is very plentiful
when we are young , but near the age of twenty - one our bodies begin to produce
less of it . ...
```

The `raw/` folder holds the original emails with full headers, HTML and attachments. It is not used by the program.

---

## 4. The theory: Naive Bayes step by step

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| $D$ | an email (document), treated as a list of words $w_1, w_2, \dots, w_n$ |
| $S$ | the class "spam" |
| $H$ | the class "ham" (Wikipedia writes this as $\neg S$, "not spam") |
| $N$ | number of training emails |
| $N_S,\ N_H$ | number of spam and ham training emails |
| $\text{count}(w, S)$ | how many times word $w$ appears in all spam training emails together |
| $T_S,\ T_H$ | total number of words in all spam / all ham training emails |
| $V$ | the vocabulary: every distinct word seen in training |
| $\lvert V \rvert$ | the size of the vocabulary (109,619 words here) |

### 4.2 Bayes' theorem

We want $P(S \mid D)$: the probability that an email is spam, given the words in it.

From the definition of conditional probability,

$$
P(D \mid S) = \frac{P(D \cap S)}{P(S)}
\qquad\text{and}\qquad
P(S \mid D) = \frac{P(D \cap S)}{P(D)}
$$

Both contain $P(D \cap S)$. Solving the first for $P(D \cap S)$ and substituting into the second gives **Bayes' theorem**:

$$
P(S \mid D) = \frac{P(S)\, P(D \mid S)}{P(D)}
$$

Each part has a name:

| Term | Symbol | Meaning in this project |
|---|---|---|
| **Prior** | $P(S)$ | probability that an email is spam *before* reading it |
| **Likelihood** | $P(D \mid S)$ | how probable these exact words are if the email is spam |
| **Evidence** | $P(D)$ | how probable these words are overall |
| **Posterior** | $P(S \mid D)$ | probability that the email is spam *after* reading it |

The evidence can be found with the law of total probability, $P(D) = P(D \mid S)\,P(S) + P(D \mid H)\,P(H)$. As section 4.7 shows, we never need to compute it.

### 4.3 The "naive" assumption

$P(D \mid S)$ is the probability of a whole email. There are far too many possible emails to estimate this directly, so Naive Bayes makes a simplifying assumption:

> **Given the class, the words of an email are independent of each other.**

With this assumption, the probability of the whole email becomes the product of the probabilities of its words:

$$
P(D \mid S) = \prod_{i=1}^{n} P(w_i \mid S) = P(w_1 \mid S) \times P(w_2 \mid S) \times \cdots \times P(w_n \mid S)
$$

and the same for ham:

$$
P(D \mid H) = \prod_{i=1}^{n} P(w_i \mid H)
$$

The assumption is called *naive* because it is clearly false. For example, "click" and "here" often appear together. It also ignores word order: the model sees an email as a "bag of words". In practice the classifier still works very well. It doesn't need exact probabilities, only to know which class is more likely.

This is the **multinomial** form of Naive Bayes. Each word in an email is treated as an independent random draw from that class's word distribution. A word that appears three times counts three times.

### 4.4 Estimating the prior

The prior is the fraction of training emails in each class:

$$
P(S) = \frac{N_S}{N}, \qquad P(H) = \frac{N_H}{N}
$$

With our training set of 22,172 emails (10,164 spam and 12,008 ham):

$$
P(S) = \frac{10164}{22172} = 0.458, \qquad P(H) = \frac{12008}{22172} = 0.542
$$

### 4.5 Estimating the word probabilities

The natural estimate of $P(w \mid S)$ is "how often $w$ appears among all the words in spam emails":

$$
\hat{P}(w \mid S) = \frac{\text{count}(w, S)}{T_S}
$$

This is the **maximum likelihood estimate** (MLE): the value that makes the training data most probable under the multinomial model. In our training set the spam emails contain $T_S = 2{,}087{,}713$ words and the ham emails contain $T_H = 3{,}300{,}607$ words.

### 4.6 The zero-probability problem and Laplace smoothing

The MLE has a serious flaw. The word **KAMINSKI** (the owner of the enron2 mailbox) appears 3,893 times in ham but **never** in spam, so

$$
\hat{P}(\text{KAMINSKI} \mid S) = \frac{0}{2087713} = 0
$$

Because the likelihood is a product, a single zero makes the whole thing zero. Any email containing "Kaminski" would get $P(D \mid S) = 0$ and could never be classified as spam, however spammy the rest of it is. In code, $\ln 0$ is also undefined and crashes the program. This is a real problem: 24,226 words in our vocabulary appear in ham but never in spam, and 64,261 appear in spam but never in ham.

The fix is **Laplace (add-one) smoothing**: pretend every vocabulary word was seen one extra time in each class.

$$
P(w \mid S) = \frac{\text{count}(w, S) + 1}{T_S + \lvert V \rvert}
\qquad\qquad
P(w \mid H) = \frac{\text{count}(w, H) + 1}{T_H + \lvert V \rvert}
$$

**Why $\lvert V \rvert$ in the denominator?** Adding 1 to the count of every one of the $\lvert V \rvert$ words adds $\lvert V \rvert$ to the total, and the denominator must grow to match so that the probabilities still add up to 1:

$$
\sum_{w \in V} \frac{\text{count}(w, S) + 1}{T_S + \lvert V \rvert}
= \frac{\left(\sum_{w \in V} \text{count}(w, S)\right) + \lvert V \rvert}{T_S + \lvert V \rvert}
= \frac{T_S + \lvert V \rvert}{T_S + \lvert V \rvert} = 1
$$

With smoothing, KAMINSKI gets a small but non-zero probability:

$$
P(\text{KAMINSKI} \mid S) = \frac{0 + 1}{2087713 + 109619} = \frac{1}{2197332} = 4.55 \times 10^{-7}
$$

It is now very strong evidence for ham, but not an automatic veto.

More generally you can add any pseudocount $\alpha > 0$ instead of 1: $P(w \mid S) = \dfrac{\text{count}(w, S) + \alpha}{T_S + \alpha \lvert V \rvert}$. In Bayesian terms, add-one smoothing is the posterior mean of the word probabilities under a uniform (Dirichlet) prior.

### 4.7 The decision rule

An email is classified as spam if spam is the more probable class given its words:

$$
\text{spam} \iff P(S \mid D) > P(H \mid D)
$$

Write both posteriors using Bayes' theorem and divide one by the other. The evidence $P(D)$ is the same in both, so it cancels:

$$
\frac{P(S \mid D)}{P(H \mid D)}
= \frac{P(S)\, P(D \mid S) \,/\, P(D)}{P(H)\, P(D \mid H) \,/\, P(D)}
= \frac{P(S)}{P(H)} \prod_{i=1}^{n} \frac{P(w_i \mid S)}{P(w_i \mid H)}
$$

This is the **posterior odds**: the **prior odds** multiplied by one **likelihood ratio** per word. A word that is 4 times more common in spam than in ham multiplies the odds of spam by 4. The email is spam if the posterior odds are greater than 1.

### 4.8 Working in log space (avoiding underflow)

Multiplying hundreds of small probabilities gives numbers too small for a computer to store. For example, the first email in our test set is a 508-word ham email. Its likelihood under the spam model is

$$
P(D \mid S) = e^{-3984.3} \approx 10^{-1730}
$$

The smallest normal number a Python float can hold is about $2.2 \times 10^{-308}$. If you multiply the 508 probabilities directly, Python returns exactly `0.0`. This is called **underflow**.

The fix is to take the natural logarithm. A logarithm turns a product into a sum, $\ln(a \times b) = \ln a + \ln b$. It is also strictly increasing, so "greater than 1" becomes "greater than 0" and the decision does not change. Taking the log of the posterior odds:

$$
\boxed{\;\ln \frac{P(S \mid D)}{P(H \mid D)} = \ln \frac{P(S)}{P(H)} + \sum_{i=1}^{n} \ln \frac{P(w_i \mid S)}{P(w_i \mid H)}\;}
$$

This is the formula the program computes. We call it the **score** $L$ of the email:

- $L > 0$: spam is more likely, so the email is classified as **spam**
- $L \le 0$: the email is classified as **ham** (a tie goes to ham)

Each word adds its own **log-likelihood ratio** to the score. Spammy words add a positive amount and hammy words subtract. For example, VIAGRA adds $+7.29$ (it is about $e^{7.29} \approx 1{,}470$ times more common in spam), while ENRON adds $-10.37$ (it is about $32{,}000$ times more common in ham).

The prior only adds $\ln(0.458 / 0.542) = -0.17$. A typical email has 110 words (median), each adding up to about $\pm 10$, so the decision is driven almost entirely by the words.

**Words never seen in training** are skipped (they add 0). If smoothing were applied to them, each unknown word would add $\ln\frac{T_H + \lvert V \rvert}{T_S + \lvert V \rvert} = \ln\frac{3410226}{2197332} = +0.44$. That would push the email towards spam only because the spam training emails contain fewer words in total, which is not real evidence.

### 4.9 Turning the score into a probability

The score is a **log-odds**, also called a *logit*. Since $P(S \mid D) + P(H \mid D) = 1$, we can recover the actual probability. Write $p = P(S \mid D)$:

$$
L = \ln \frac{p}{1 - p}
\quad\Longrightarrow\quad
\frac{p}{1 - p} = e^{L}
\quad\Longrightarrow\quad
p = \frac{e^{L}}{1 + e^{L}} = \frac{1}{1 + e^{-L}}
$$

This is the **logistic (sigmoid) function**. It maps any score to a probability between 0 and 1, and a score of 0 maps to exactly 50%. The program uses it to print `P(spam | message)` for messages you type.

`math.exp()` fails if its argument is larger than about 709.78, because the result would exceed the largest float ($\approx 1.8 \times 10^{308}$). Very hammy emails can have $L < -709$, so the code uses `max(score, -700)`. At that point the probability is about $10^{-304}$, which prints as 0.00% anyway.

---

## 5. Worked example

Let's classify the two-word message **"free money"** by hand, using the counts from the training set.

**Step 1: the counts and the smoothed probabilities.** The denominators are $T_S + \lvert V \rvert = 2{,}087{,}713 + 109{,}619 = 2{,}197{,}332$ for spam and $T_H + \lvert V \rvert = 3{,}300{,}607 + 109{,}619 = 3{,}410{,}226$ for ham.

| Word | count in spam | count in ham | $P(w \mid S)$ | $P(w \mid H)$ | $\ln \dfrac{P(w \mid S)}{P(w \mid H)}$ |
|---|---:|---:|---|---|---:|
| FREE | 2,982 | 1,142 | $\frac{2983}{2197332} = 0.0013576$ | $\frac{1143}{3410226} = 0.00033517$ | +1.3988 |
| MONEY | 3,683 | 975 | $\frac{3684}{2197332} = 0.0016766$ | $\frac{976}{3410226} = 0.00028620$ | +1.7678 |

**Step 2: add up the score.**

$$
\begin{aligned}
L &= \ln\frac{P(S)}{P(H)} + \ln\frac{P(\text{FREE} \mid S)}{P(\text{FREE} \mid H)} + \ln\frac{P(\text{MONEY} \mid S)}{P(\text{MONEY} \mid H)} \\[4pt]
  &= \ln\frac{10164}{12008} + 1.3988 + 1.7678 \\[4pt]
  &= -0.1667 + 1.3988 + 1.7678 = 3.00
\end{aligned}
$$

$L > 0$, so the message is classified as **spam**.

**Step 3: convert to a probability.**

$$
P(S \mid \text{"free money"}) = \frac{1}{1 + e^{-3.00}} = \frac{1}{1 + 0.0498} = 0.9526 = 95.26\%
$$

This matches the program's output: `-> SPAM  (P(spam | message) = 95.26%)`.

**The same calculation as odds.** The prior odds of spam are $10164 / 12008 = 0.846$. FREE is $e^{1.3988} = 4.05$ times more common in spam and MONEY is $e^{1.7678} = 5.86$ times more common, so the posterior odds are $0.846 \times 4.05 \times 5.86 = 20.1$. In other words, about 20 to 1 in favour of spam, and $20.1 / 21.1 = 95.26\%$.

---

## 6. How the code works

### Formula to code

| Formula | Code in [spam_filter.py](spam_filter.py) |
|---|---|
| $P(S) = N_S / N$ | `self.p_spam = n_spam / len(emails)` |
| $P(w \mid S) = \dfrac{\text{count}(w,S) + 1}{T_S + \lvert V \rvert}$ | `p_word_spam = (spam_counts[word] + 1) / (spam_total + len(vocab))` |
| $\ln \dfrac{P(w \mid S)}{P(w \mid H)}$ for every word | `self.word_log_ratio[word] = math.log(p_word_spam / p_word_ham)` |
| $L = \ln \dfrac{P(S)}{P(H)} + \sum_i \ln \dfrac{P(w_i \mid S)}{P(w_i \mid H)}$ | the `score()` method |
| spam if $L > 0$ | the `predict()` method |
| $P(S \mid D) = \dfrac{1}{1 + e^{-L}}$ | `1 / (1 + math.exp(-max(score, -700)))` |

### `tokenize(text)`: splitting an email into words

```python
return re.findall(r"[A-Z]+", text.upper())
```

The text is converted to uppercase, and every run of the letters A–Z becomes one word. Numbers, punctuation and symbols are dropped, so `"Win $1,000 NOW!!!"` becomes `["WIN", "NOW"]`. Uppercasing makes "free", "Free" and "FREE" the same word.

### `load_emails()`: reading the data

```python
for label in ["spam", "ham"]:
    for path in sorted(DATA_DIR.glob(f"enron*/{label}/*.txt")):
        emails.append((path.read_text(encoding="latin-1"), label))
```

This finds every `.txt` file in `data/preprocessed/enron*/spam/` and `.../ham/`. The folder name gives the label. The result is a list of `(text, label)` pairs.

- `latin-1` decoding can read any byte, so a few unusual characters in old emails never crash the program.
- `sorted()` loads the files in a fixed order. Together with the fixed random seed, this makes the train/test split the same on every run.

### `NaiveBayesSpamFilter.train(emails)`: learning the probabilities

Training is only counting, done in three steps:

1. **Count the words.** Go through every training email and add its words to `spam_counts` or `ham_counts`. These are `Counter` objects, dictionaries that map each word to how many times it was seen. The emails in each class are also counted, as `n_spam` and `n_ham`.
2. **Priors** (section 4.4): `p_spam = n_spam / len(emails)` and `p_ham = n_ham / len(emails)`.
3. **Word log-ratios** (sections 4.6 and 4.8): for every word in the vocabulary, compute the two smoothed probabilities and store $\ln\frac{P(w \mid S)}{P(w \mid H)}$ in the dictionary `word_log_ratio`.

Only the log-ratio of each word is kept, because that is all the decision rule needs. Since they are computed once during training, classifying an email is just a sum of dictionary lookups.

### `score(text)` and `predict(text)`: classifying

```python
score = math.log(self.p_spam / self.p_ham)        # prior log-odds
for word in tokenize(text):
    score += self.word_log_ratio.get(word, 0)     # + one log-likelihood ratio per word
```

`score()` is the boxed formula from section 4.8. `.get(word, 0)` makes words that never appeared in training add 0, so they are ignored. `predict()` returns `"spam"` if the score is greater than 0 and `"ham"` otherwise.

### `main()`: putting it together

1. Load all emails and count spam and ham.
2. Shuffle with `random.seed(42)` and split 80/20 into a training set and a test set.
3. Train the model and print the priors and the vocabulary size.
4. Predict every test email, count the four outcomes (section 7) and print the metrics.
5. Sort the vocabulary by log-ratio and print the 10 most spammy and 10 most hammy words.
6. Ask you for messages to classify, printing the label and $P(\text{spam} \mid \text{message})$.

---

## 7. Evaluation method

### Train/test split

A model must be tested on emails it has **not** seen during training. Otherwise we would only measure how well it memorised the training data. The 27,716 emails are shuffled randomly and split:

| Set | Emails | Spam | Ham | Used for |
|---|---:|---:|---:|---|
| Training (80%) | 22,172 | 10,164 | 12,008 | counting words and estimating all probabilities |
| Test (20%) | 5,544 | 2,507 | 3,037 | measuring performance only |

### Confusion matrix

Every test email ends up in one of four cells, with spam as the "positive" class:

| | Predicted ham | Predicted spam |
|---|---:|---:|
| **Actually ham** | **TN** (true negative): ham correctly delivered | **FP** (false positive): ham wrongly sent to spam |
| **Actually spam** | **FN** (false negative): spam that got through | **TP** (true positive): spam correctly caught |

### Metrics

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\qquad \text{the fraction of all emails classified correctly}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
\qquad \text{of the emails marked as spam, the fraction that really are spam}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
\qquad \text{of all spam emails, the fraction that were caught}
$$

The two kinds of mistake do not cost the same. A **false positive** means a real email, perhaps an important one, disappears into the spam folder. A **false negative** only means one extra junk email in the inbox. That is why precision, and the false positive rate $\frac{FP}{FP + TN}$ (the fraction of real emails wrongly flagged), matter a lot for a spam filter.

---

## 8. Results

### Performance on the test set

| | Predicted ham | Predicted spam | Total |
|---|---:|---:|---:|
| **Actually ham** | TN = 3,004 | FP = 33 | 3,037 |
| **Actually spam** | FN = 26 | TP = 2,481 | 2,507 |

| Metric | Calculation | Value |
|---|---|---:|
| Accuracy | $(2481 + 3004) \,/\, 5544$ | **98.94%** |
| Precision | $2481 \,/\, (2481 + 33)$ | **98.69%** |
| Recall | $2481 \,/\, (2481 + 26)$ | **98.96%** |
| False positive rate (from the matrix, not printed by the program) | $33 \,/\, (33 + 3004)$ | **1.09%** |

Only 59 of the 5,544 test emails were misclassified. About 1 in 92 real emails would be wrongly sent to the spam folder, and about 1 in 96 spam emails would get through.

### The most telling words

These are the words with the largest (spam) and smallest (ham) log-ratio $\ln\frac{P(w \mid S)}{P(w \mid H)}$:

- **Spam:** VIAGRA, VOIP, PILLS, CIALIS, OOKING, WIIL, WYSAK, UR, PHOTOSHOP, NBSP
  - drug adverts (VIAGRA, CIALIS, PILLS) and cheap software or services (PHOTOSHOP, VOIP)
  - misspellings and junk tokens (OOKING, WIIL, WYSAK, UR) that spammers use to slip past filters
  - NBSP, left over from the HTML code `&nbsp;`, because spam is often sent as formatted HTML
- **Ham:** HPL, ENRONXGATE, FASTOW, MMBTU, ENA, EES, ECT, DYNEGY, KAMINSKI, ENRON
  - all of these are Enron's internal vocabulary: the company name, its divisions (ECT, ENA, EES), its email system (ENRONXGATE), people (Fastow was the CFO, Kaminski the owner of the enron2 mailbox), business partners (Dynegy) and gas-trading units (MMBTU)
  - ENRON appears 49,719 times in ham training emails and never once in spam

The classifier has learned that "talks about Enron business" means legitimate. That works very well for Enron's mailboxes, but it also shows that **the filter is specialised to this company** (see section 9).

### Trying your own messages

| Message | Prediction | P(spam \| message) |
|---|---|---:|
| Congratulations! You won a FREE prize, click here to claim your cash now | SPAM | 100.00% |
| Hi Vince, can we move tomorrow's meeting about the gas contract to 3pm? | HAM | 0.00% |
| free money | SPAM | 95.26% |
| limited time offer | SPAM | 73.25% |
| please see attached report | HAM | 12.39% |

Short, ambiguous messages get in-between probabilities. Longer messages quickly reach 100% or 0%, as section 9 explains.

---

## 9. Limitations and possible improvements

### Limitations

1. **The independence assumption is false, so the probabilities are overconfident.** Words like "click", "here" and "claim" tend to appear together, but the model treats each as separate evidence and counts the same signal several times. Classification is still very accurate, but the printed probabilities often show 100.00% or 0.00%. They should be read as "very sure", not as exact probabilities.
2. **Company-specific vocabulary.** The strongest ham signals are Enron names and jargon. On a different person's inbox the filter would lose many of its best clues and perform worse until retrained on their own emails.
3. **Ham and spam come from different sources and years.** In the subsets used here, all ham dates from 1999–2002 and comes from Enron mailboxes, while spam dates from 2001–2005 and comes from outside collections. Part of what the model learns may be "old Enron email vs. newer email" rather than "legitimate vs. spam", so real-world accuracy would probably be lower.
4. **Random split instead of a time-ordered split.** The original paper trains on older emails and tests on newer ones, as a real filter would. A random split is easier, because near-identical spam emails can end up in both the training and the test set.
5. **Bag of words.** Word order and phrases are ignored ("not free" looks like "free"), and so are numbers, punctuation and words never seen in training.
6. **A fixed 50% threshold.** Both kinds of mistake are treated as equally bad, even though losing a real email is worse.

### Possible improvements

- **Tune the threshold:** classify as spam only if $P(S \mid D) > \lambda$ for a high $\lambda$, such as 0.99. This is the same as requiring $L > \ln\frac{\lambda}{1 - \lambda}$, and it trades a few more missed spam emails for fewer lost real emails.
- **k-fold cross-validation:** repeat the train/test split several times and average the results, to get a more reliable accuracy estimate with a confidence interval.
- **Train on some subsets and test on others** (for example, train on enron1–3 and test on enron5–6) to measure how well the filter generalises to new mailboxes.
- **Try other Naive Bayes variants,** such as Bernoulli Naive Bayes (did the word appear or not), which the original paper compares.
- **Richer features:** word pairs (bigrams), keeping numbers and symbols like `$` and `!`, or removing very common words ("the", "and").

---

## 10. References

1. Wikipedia, *Naive Bayes classifier*, especially the sections "Multinomial naive Bayes" and "Document classification": <https://en.wikipedia.org/wiki/Naive_Bayes_classifier>
2. V. Metsis, I. Androutsopoulos and G. Paliouras, *"Spam Filtering with Naive Bayes – Which Naive Bayes?"*, Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.
3. The Enron-Spam datasets: <https://www2.aueb.gr/users/ion/data/enron-spam/>
