import os
import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"\S+", text.upper())


def load_email(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        return file.read()


def count_files(folder_path):
    return sum(len(files) for _, _, files in os.walk(folder_path))


def train_counts(folder_path):
    token_counts = Counter()
    email_count = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            token_counts.update(tokenize(load_email(file_path)))
            email_count += 1

    return token_counts, email_count



def score_email(
    email_counts,
    class_counts,
    class_token_total,
    log_class_prior,
    vocabulary,
    alpha=1,
):
    score = log_class_prior
    vocabulary_size = len(vocabulary)
    denominator = class_token_total + alpha * vocabulary_size

    for token, count in email_counts.items():
        if token not in vocabulary:
            continue

        token_probability = (class_counts.get(token, 0) + alpha) / denominator
        score += count * math.log(token_probability)

    return score


def print_formulas():
    print("""
'''
Formulas Used
    C = class, either ham or spam
    w = one token/word from the email

    P(C) = number of training emails in class C / total training emails
    P(w | C) = (count(w in class C) + 1) / (total tokens in class C + vocabulary size)

    score(C) = log(P(C)) + sum(count(w in email) * log(P(w | C)))

    Predict spam if score(spam) > score(ham), otherwise predict ham.

    Laplace smoothing:
        We add 1 to every word count.
        This means a word that did not appear in one class still gets a tiny probability.

    Out-of-vocabulary words:
        If a word never appeared in training at all, this code ignores it.
'''
""")


train_path = "data/preprocessed/enron1"
test_path = "data/preprocessed/enron2"

train_ham_path = os.path.join(train_path, "ham")
train_spam_path = os.path.join(train_path, "spam")
test_ham_path = os.path.join(test_path, "ham")
test_spam_path = os.path.join(test_path, "spam")

print(f"Training files from: {train_path}")
print(f"Testing files from: {test_path}")
print()
print_formulas()

ham_total_counts, ham_email_count = train_counts(train_ham_path)
spam_total_counts, spam_email_count = train_counts(train_spam_path)

ham_token_total = sum(ham_total_counts.values())
spam_token_total = sum(spam_total_counts.values())
total_train_emails = ham_email_count + spam_email_count
vocabulary = set(ham_total_counts) | set(spam_total_counts)
vocabulary_size = len(vocabulary)
alpha = 1

ham_prior = ham_email_count / total_train_emails
spam_prior = spam_email_count / total_train_emails
log_ham_prior = math.log(ham_prior)
log_spam_prior = math.log(spam_prior)

example_word = "FREE"
example_ham_count = ham_total_counts.get(example_word, 0)
example_spam_count = spam_total_counts.get(example_word, 0)
example_ham_probability = (example_ham_count + alpha) / (
    ham_token_total + alpha * vocabulary_size
)
example_spam_probability = (example_spam_count + alpha) / (
    spam_token_total + alpha * vocabulary_size
)

print("""
'''
Training Counts
    First, the program reads the training folders:
        data/preprocessed/enron1/ham
        data/preprocessed/enron1/spam

    Then it counts:
        How many ham emails?
        How many spam emails?
        How many total tokens are in ham?
        How many total tokens are in spam?
        How many unique tokens are in the vocabulary?
'''
""")
print(f"Ham emails: {ham_email_count}")
print(f"Spam emails: {spam_email_count}")
print(f"Ham tokens: {ham_token_total}")
print(f"Spam tokens: {spam_token_total}")
print(f"Vocabulary size: {vocabulary_size}")
print()

print(f"""
'''
Class Priors
    The class prior means:
        Before looking at the words, how likely is an email to be ham or spam?

    P(C) = number of training emails in class C / total training emails

    P(ham) = {ham_email_count} / {total_train_emails}
    P(spam) = {spam_email_count} / {total_train_emails}

    The code stores log(P(ham)) and log(P(spam)) because we add log probabilities.
'''
""")
print(f"log P(ham): {log_ham_prior}")
print(f"log P(spam): {log_spam_prior}")
print()
print()

print(f"""
'''
Word Probabilities
    How likely is this word if the email is spam?
    How likely is this word if the email is ham?

    P(w | C) = (count(w in class C) + 1) / (total tokens in class C + vocabulary size)

    Example with the word {example_word}:
        count({example_word} in spam) = {example_spam_count}
        count({example_word} in ham) = {example_ham_count}

        P({example_word} | spam) = ({example_spam_count} + 1) / ({spam_token_total} + {vocabulary_size})
        P({example_word} | ham)  = ({example_ham_count} + 1) / ({ham_token_total} + {vocabulary_size})
'''
""")
print(f"P({example_word} | spam): {example_spam_probability}")
print(f"P({example_word} | ham): {example_ham_probability}")

print()
print("""
'''
Scoring An Email
    score(C) = log(P(C)) + sum(count(w in email) * log(P(w | C)))
    Example email:
        FREE MONEY FREE
        FREE: 2 MONEY: 1

        The spam score is:
            log(P(spam))
            + 2 * log(P(FREE | spam))
            + 1 * log(P(MONEY | spam))
        
        The ham score is:
            log(P(ham))
            + 2 * log(P(FREE | ham))
            + 1 * log(P(MONEY | ham))

    Predict spam if score(spam) > score(ham), otherwise predict ham
'''
""")
actual_ham_count = count_files(test_ham_path)
actual_spam_count = count_files(test_spam_path)

predicted_ham_count = 0
predicted_spam_count = 0
ham_correct = 0
ham_wrong = 0
spam_correct = 0
spam_wrong = 0

for actual_label, folder_path in [("ham", test_ham_path), ("spam", test_spam_path)]:
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            email_counts = Counter(tokenize(load_email(file_path)))

            ham_score = score_email(
                email_counts,
                ham_total_counts,
                ham_token_total,
                log_ham_prior,
                vocabulary,
                alpha,
            )
            spam_score = score_email(
                email_counts,
                spam_total_counts,
                spam_token_total,
                log_spam_prior,
                vocabulary,
                alpha,
            )

            predicted_label = "spam" if spam_score > ham_score else "ham"

            if predicted_label == "spam":
                predicted_spam_count += 1
            else:
                predicted_ham_count += 1

            if actual_label == "ham" and predicted_label == "ham":
                ham_correct += 1
            elif actual_label == "ham":
                ham_wrong += 1
            elif predicted_label == "spam":
                spam_correct += 1
            else:
                spam_wrong += 1

total_test_emails = actual_ham_count + actual_spam_count
accuracy = (ham_correct + spam_correct) / total_test_emails

print(f"""
'''
Testing
    Now the program reads the test folders:
        data/preprocessed/enron2/ham
        data/preprocessed/enron2/spam

    These emails were not used for training.
    The model predicts each one as ham or spam, then compares the prediction to the real folder.
'''
""")

print("Actual test counts")
print(f"Spam count: {actual_spam_count}")
print(f"Ham count: {actual_ham_count}")
print()

print("Predicted test counts")
print(f"Predicted spam count: {predicted_spam_count}")
print(f"Predicted ham count: {predicted_ham_count}")
print()

print("Confusion matrix")
print("              predicted ham  predicted spam")
print(f"actual ham    {ham_correct:>13}  {ham_wrong:>14}")
print(f"actual spam   {spam_wrong:>13}  {spam_correct:>14}")
print()
print(f"""
'''
Meaning of confusion matrix:
    {ham_correct} ham emails were correctly predicted as ham
    {ham_wrong} ham emails were wrongly predicted as spam
    {spam_wrong} spam emails were wrongly predicted as ham
    {spam_correct} spam emails were correctly predicted as spam
'''
""")

print("Accuracy: correct predictions / total emails")
print(f"Accuracy: {accuracy:.2%}")
