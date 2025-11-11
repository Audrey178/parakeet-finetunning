import string 
import re 

delete_chars = string.punctuation + '\n'
table = str.maketrans(delete_chars, ' ' * len(delete_chars))

def format_string(text: string):
    text = text.replace("<unk>", " ").replace("'", " ")
    return re.sub(r'\s+', ' ', text.translate(table).lower().strip())


