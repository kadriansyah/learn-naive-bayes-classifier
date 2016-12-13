import numpy as np
import pandas as pd
import os

email_path  = "emails/" # Eg. /Users/Daniel/Emails/
email_files = [x for x in os.listdir(email_path)] #creates list of email filenames

to_filter = [',', '?', '!', '-', '>', '$', '.']

prob_spam_email_file = 0.5 # 3 spam emails divide by total emails
prob_nonspam_email_file = 0.5 # 3 non-spam emails divide by total emails

def parse_files(email_files, email_path):
    emails = []
    labels = [] # non-spam = 0, spam = 1
    for e in email_files:
        with open(email_path + e, 'r') as email:
            text = [x for x in email.read().lower().replace('\n', ' ')]
            for ch in to_filter:
                text = [x.replace(ch, ' ') for x in text]

            text = ''.join(text).split()
            label = text[-1]
            labels.append(label)
            del text[-1] # so it doesn't try use the label in predictions
            emails.append(text)
    return emails,labels

def create_frequency_table(texts, labels=None, parse=False):
    freq_table = pd.DataFrame([])
    for idx, t in enumerate(texts):
        vocab = set(t) # remove duplicate word
        d = pd.Series({ v : t.count(v) for v in vocab})

        if labels != None:
            d['*class*'] = labels[idx] # to make sure class column appears on the left when we print freq_table

        freq_table = freq_table.append(d, ignore_index=True)

    return freq_table.fillna(0)

def train(frequency_table):
    frequencies = frequency_table.iloc[:, 1:]
    labels = frequency_table.iloc[:, 0].values
    vocab = list(frequencies.columns.values) # word vocabulary

    spam, nonspam = pd.DataFrame([]), pd.DataFrame([])
    for idx, row in frequencies.iterrows():
        if labels[idx] == '1':
            spam = spam.append(row)
        else:
            nonspam = nonspam.append(row)

    spam_probs, nonspam_probs = {}, {}
    spam_word_count = sum([word for word in spam.sum()]) # spam.sum() is sum per column in spam DataFrame (it means total sum per word)
    nonspam_word_count = sum([word for word in nonspam.sum()]) # nonspam.sum() is sum per column in non-spam DataFrame

    alpha = 1 # Laplace Smoothing
    for word in vocab:
        word_occurences_spam = spam[word].sum() # column sum
        word_occurences_nonspam = nonspam[word].sum() # column sum

        # remember: P(w|c) = ( count(w,c) + 1 ) / ( count(c) + |V| )
        bayesian_prob_spam = (word_occurences_spam + alpha) / (spam_word_count + len(vocab))
        bayesian_prob_nonspam = (word_occurences_nonspam + alpha) / (spam_word_count + len(vocab))

        spam_probs[word], nonspam_probs[word] = bayesian_prob_spam, bayesian_prob_nonspam

    return spam_probs, nonspam_probs

def predict(text, spam_prob, nonspam_prob):
    text = [x for x in text.replace('\n', ' ')]
    for ch in to_filter:
        text = [x.replace(ch, ' ') for x in text]

    prsd_text = [''.join(text).split()]

    txt_table = create_frequency_table(texts=prsd_text)
    vocab = txt_table.columns.values

    spam_likelihood = 1
    nonspam_likelihood = 1
    for word in vocab:
        if word in spam_prob:
            spam_likelihood *= spam_prob[word]

        if word in nonspam_prob:
            nonspam_likelihood *= nonspam_prob[word]

    prob_spam_given_text = prob_spam_email_file * spam_likelihood
    prob_nonspam_given_text = prob_nonspam_email_file * nonspam_likelihood

    return int((prob_spam_given_text / prob_nonspam_given_text) >= 1)

def make_prediction(text):
    emails, labels = parse_files(email_files, email_path)

    del labels[0]
    del emails[0] # I had to do this because of quirk with txt file

    freq_tbl = create_frequency_table(emails, labels)
    nb_spam, nb_nonspam = train(freq_tbl)

    return predict(text, nb_spam, nb_nonspam)
