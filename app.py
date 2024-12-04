from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import json
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the Ecommerce FAQ Chatbot dataset
with open('Ecommerce_FAQ_Chatbot_dataset.json', 'r') as f:
    faq_data = json.load(f)

from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare the dataset by encoding all FAQ questions
faq_list = faq_data['questions']
faq_questions = [item['question'] for item in faq_list]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

def find_answer(user_question):
    # Encode the user's question
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Compute cosine similarities between the user question and the FAQ questions
    similarities = util.pytorch_cos_sim(question_embedding, faq_embeddings)
    
    # Find the FAQ with the highest similarity score
    best_match_idx = similarities.argmax().item()
    best_answer = faq_list[best_match_idx]['answer']
    
    return best_answer

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.form['question']  # Get the user’s question from the form
    answer = find_answer(user_question)  # Find the answer to the question using the transformer
    return jsonify({'answer': answer})  # Return the answer as JSON

if __name__ == '__main__':
    app.run(debug=True)
