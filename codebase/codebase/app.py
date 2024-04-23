from flask import Flask, request, jsonify
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
import re

# Search() is defined to interact with the search system developed earlier
from text_indexer_and_search import search

app = Flask(__name__)

# Download necessary NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def spell_check_and_suggest(word):
    # Simple implementation using NLTK's WordNet
    if wordnet.synsets(word):
        return word  # No correction needed
    else:
        # Generate suggestions (This is a placeholder; consider using a more robust method)
        suggestions = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                suggestions.add(lemma.name())
        return suggestions

def expand_query(query):
    expanded_terms = []
    tokens = word_tokenize(query)
    for token in tokens:
        synonyms = set()
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        expanded_terms.append(list(synonyms))
    expanded_query = ' '.join([item for sublist in expanded_terms for item in sublist])
    return expanded_query if expanded_query else query

@app.route('/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    
    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    query = data['query']
    k = int(data.get('k', 3))  # Default top-k results

    # Optional: Spell correction
    corrected_words = [spell_check_and_suggest(word) for word in query.split()]
    corrected_query = ' '.join(corrected_words)

    # Optional: Query expansion
    expanded_query = expand_query(corrected_query)
    
    top_docs, similarities = search(expanded_query)

    results = []
    for i in range(min(k, len(top_docs))):
        results.append({'doc_id': top_docs[i], 'similarity': similarities[i]})
    
    return jsonify({'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
