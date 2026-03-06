
from nltk.corpus import wordnet

def expand_feature(playlist_name):
    '''
    Uses WordNet to expand the features of a playlist name by finding 
    synonyms and hypernyms, returning a space-separated string for TF-IDF    '''

    #gandle empty playlist names
    if not isinstance(playlist_name, str): 
        return ""
    
    expanded_features = set()

    for word in playlist_name.split():
        # add the original word
        expanded_features.add(word)

        # get all synsets for the word
        for syn in wordnet.synsets(word):
            # add synonyms (lemmas)
            for lemma in syn.lemmas():
                expanded_features.add(lemma.name().replace('_', ' '))

            # add hypernyms (more general terms)
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    expanded_features.add(lemma.name().replace('_', ' '))
    
    return ' '.join(expanded_features)


test_name = "chill grill"
print(f"Original: {test_name}")
print(f"Expanded: {expand_feature(test_name)}")

