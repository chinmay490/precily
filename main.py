from flask import Flask, jsonify, redirect, url_for, render_template
from flask import request
from sentence_transformers import  util
import pickle

app = Flask(__name__)
similarity_model = pickle.load(open('perline_model.sav' , 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        te1 = request.form["text1"]
        te2 = request.form["text2"]
        embedding1 = similarity_model.encode(te1, convert_to_tensor=True)
        embedding2 = similarity_model.encode(te2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

        return {'similarity score' : cosine_scores.item()}  
    return render_template("index.html")

@app.route('/score', methods=["POST"])
def pjson():
    data = request.get_json()
    te1 = data["text1"]
    te2 = data["text2"]
    embedding1 = similarity_model.encode(te1, convert_to_tensor=True)
    embedding2 = similarity_model.encode(te2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

    return {'similarity score' : cosine_scores.item()} 



if __name__ == "__main__":
    app.run(debug=True)




'''
@app.post('/score')
def smiliarity_score(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    te1 = input_dict['text1']
    te2 = input_dict['text2']

    embedding1 = similarity_model.encode(te1, convert_to_tensor=True)
    embedding2 = similarity_model.encode(te2, convert_to_tensor=True)
'''