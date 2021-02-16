import string
import re
def preprocess_hindi(sent):
   
    sent = str(sent)
    sent = sent.strip()
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)
    sent = re.sub(r"[२३०८१५७९४६]","",sent)
    sent = (sent.translate(str.maketrans('', '', string.punctuation))).replace('।','')
    sent = ''.join([i for i in sent if not i.isdigit()])
    sent = sent.rstrip().strip()
    sent = '<s> ' + sent + ' <e>'
    return sent

def preprocess_eng(sent):

    sent = str(sent)
    sent = sent.lower()
    sent = re.sub("'", '', sent)
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)
    sent = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sent)
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent = ''.join([i for i in sent if not i.isdigit()])
    sent = sent.rstrip().strip()
    sent = '<s> ' + sent + ' <e>'
    return sent