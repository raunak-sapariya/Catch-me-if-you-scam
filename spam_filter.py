"""
Email spam filter using a Naive Bayes classifier.

Dataset: Enron-Spam (preprocessed)  https://www2.aueb.gr/users/ion/data/enron-spam/
Theory : https://en.wikipedia.org/wiki/Naive_Bayes_classifier  (section "Document classification")

Bayes' theorem:
    P(spam | email) = P(spam) * P(email | spam) / P(email)

"Naive" assumption - the words of an email are independent given the class:
    P(email | spam) = P(w1 | spam) * P(w2 | spam) * ... * P(wn | spam)

Dividing P(spam | email) by P(ham | email) cancels P(email). Taking the log turns the
product of many tiny probabilities (which would round down to 0) into a sum:
    ln[P(spam|email) / P(ham|email)] = ln[P(spam) / P(ham)] + sum of ln[P(wi|spam) / P(wi|ham)]

If this value is greater than 0, spam is more likely than ham, so the email is spam.
"""

import math
import random
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "preprocessed"


def tokenize(text):
    return re.findall(r"[A-Z]+", text.upper())


def load_emails():
    emails = []
    for label in ["spam", "ham"]:
        for path in sorted(DATA_DIR.glob(f"enron*/{label}/*.txt")):
            emails.append((path.read_text(encoding="latin-1"), label))
    return emails


class NaiveBayesSpamFilter:
    def train(self, emails):
        """Estimate P(spam), P(ham) and P(word | class) by counting words."""
        spam_counts = Counter()  # how many times each word appears in spam emails
        ham_counts = Counter()   # how many times each word appears in ham emails
        n_spam = n_ham = 0
        for text, label in emails:
            if label == "spam":
                spam_counts.update(tokenize(text))
                n_spam += 1
            else:
                ham_counts.update(tokenize(text))
                n_ham += 1

        """
        Estimate the prior probabilities of spam and ham.
        """
        self.p_spam = n_spam / len(emails)
        self.p_ham = n_ham / len(emails)

        """
        Estimate the likelihood of each word given spam or ham, using Laplace smoothing.
        This prevents any word from having a zero probability.
            P(word | spam) = (count(word in spam) + 1) / (total words in spam + vocabulary size)
            P(word | ham) = (count(word in ham) + 1) / (total words in ham + vocabulary size)
        The log ratio ln[P(word|spam) / P(word|ham)] is stored for each word.
        """
        vocab = set(spam_counts) | set(ham_counts)
        spam_total = sum(spam_counts.values())
        ham_total = sum(ham_counts.values())
        self.word_log_ratio = {}  # ln[P(word|spam) / P(word|ham)] for every word
        for word in vocab:
            p_word_spam = (spam_counts[word] + 1) / (spam_total + len(vocab))
            p_word_ham = (ham_counts[word] + 1) / (ham_total + len(vocab))
            self.word_log_ratio[word] = math.log(p_word_spam / p_word_ham)

    def score(self, text):
        """ln[P(spam|text) / P(ham|text)]. Positive means spam is more likely."""
        score = math.log(self.p_spam / self.p_ham)
        for word in tokenize(text):
            score += self.word_log_ratio.get(word, 0)  # words never seen in training are ignored
        return score

    def predict(self, text):
        return "spam" if self.score(text) > 0 else "ham"


def main():
    emails = load_emails()
    n_spam = sum(1 for _, label in emails if label == "spam")
    print(f"Loaded {len(emails)} emails: {n_spam} spam, {len(emails) - n_spam} ham")

    """
    Split into training and test sets (80% / 20%)
    """
    random.seed(42)  # fixed seed so the split (and the results) are the same every run
    random.shuffle(emails)
    split = int(0.8 * len(emails))
    train_set, test_set = emails[:split], emails[split:]
    print(f"Training on {len(train_set)} emails, testing on {len(test_set)} emails\n")

    """
    Train the model and print some statistics
    """
    model = NaiveBayesSpamFilter()
    model.train(train_set)
    print(f"Prior probabilities: P(spam) = {model.p_spam:.3f}, P(ham) = {model.p_ham:.3f}")
    print(f"Vocabulary size: {len(model.word_log_ratio)} words\n")

    """
    Test: count (actual, predicted) pairs to build the confusion matrix
    """
    results = Counter((label, model.predict(text)) for text, label in test_set)
    tp = results[("spam", "spam")]  # spam correctly caught
    fn = results[("spam", "ham")]   # spam that got through
    fp = results[("ham", "spam")]   # ham wrongly marked as spam
    tn = results[("ham", "ham")]    # ham correctly let through

    print(f"Accuracy : {(tp + tn) / len(test_set):.2%}")
    print(f"Precision: {tp / (tp + fp):.2%}  (of emails marked spam, how many really are spam)")
    print(f"Recall   : {tp / (tp + fn):.2%}  (of all spam emails, how many were caught)")
    print("\nConfusion matrix:")
    print("              predicted ham  predicted spam")
    print(f"actual ham    {tn:>13}  {fp:>14}")
    print(f"actual spam   {fn:>13}  {tp:>14}")

    """
    Words with the biggest ln[P(word|spam) / P(word|ham)] are the strongest spam signals
    """
    ranked = sorted(model.word_log_ratio, key=model.word_log_ratio.get, reverse=True)
    print("\nTop spam words:", ", ".join(ranked[:10]))
    print("Top ham words :", ", ".join(ranked[-10:]))

    """
    Try your own messages:
    The probability of being spam given a message is calculated from the score using the logistic function:
        P(spam | message) = 1 / (1 + e^(-score))
    """
    while True:
        message = input("\nType a message to classify (or press Enter to quit): ")
        if not message:
            break
        score = model.score(message)
        p_spam = 1 / (1 + math.exp(-max(score, -700)))  # max() keeps exp() from overflowing
        print(f"-> {model.predict(message).upper()}  (P(spam | message) = {p_spam:.2%})")


if __name__ == "__main__":
    main()
