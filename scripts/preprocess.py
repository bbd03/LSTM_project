import re
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, lowercase=False, remove_punctuation=True, replace_numbers=False, remove_stopwords=False, custom_stopwords=None):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.replace_numbers = replace_numbers
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def _lowercase(self, text):
        return text.lower() if self.lowercase else text

    def _remove_punctuation(self, text):
        return re.sub(r'[^\w\s<>\?]', '', text) if self.remove_punctuation else text
    
    def _replace_numbers(self, text):
        return re.sub(r'\d+', '<NUM>', text) if self.replace_numbers else text
    
    def _remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]
    
    def preprocess(self, text):
        text = self._lowercase(text)
        text = self._remove_punctuation(text)
        text = self._replace_numbers(text)
        
        tokens = text.split()
        
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        
        return tokens

# Usage example
if __name__ == "__main__":

    preprocessor = TextPreprocessor()

    sample_text = "i don't fucking hate the niggers they are awful!! how dare u "
    processed_tokens = preprocessor.preprocess(sample_text)
    print(processed_tokens)  
