from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import os

app = Flask(__name__)

# Load FAQs
with open('faqs.json', 'r') as file:
    data = json.load(file)

que_and_ans = {}
for cat in data:
    for i in data[cat]:
        que_and_ans[i['question']] = i['answer']

faqs = list(que_and_ans.keys())

# Check if the model.pkl file exists
if os.path.exists("model.pkl"):
    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
else:
    # Load the model if the file does not exist
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Save the model to model.pkl for future use
    with open("model.pkl", 'wb') as f:
        pickle.dump(model, f)
faq_embeddings = model.encode(faqs)



@app.route('/')
def index():
    return render_template('searchbox.html')

@app.route('/search')
def search():
    user_input = request.args.get('query')
    user_input_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_input_embedding, faq_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    
    top_faqs = [faqs[i] for i in top_indices]
    return jsonify({'faqs': top_faqs})

@app.route('/answer')
def answer():
    question = request.args.get('question')
    answer = que_and_ans.get(question, "Sorry, no answer found.")
    print(f"Question: {question}, Answer: {answer}")  # Print for debugging
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

