import json

from flask import Flask, jsonify, request
import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type
from flask import Flask
import logging
from logging.handlers import RotatingFileHandler




app = Flask(__name__)
model = None


handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)
	return model

modelpath = "pipeline.dill"
load_model(modelpath)


@app.route('/')
def index():
    return '''
                <a href=http://127.0.0.1:5000/prediction style="font-size: 50px">Start prediction (press the link)</a>'''



@app.route('/prediction', methods=['GET', 'POST'])

def form_example():
    # handle the POST request
    if request.method == 'POST':
        data = request.form.get('comment')
        comment = {"Comment": data}
        comment = json.dumps(comment)
        comment = json.loads(comment)
        if comment["Comment"]:
           comment = comment['Comment']
        with open('pipeline.dill', 'rb') as in_strm:

        	model = dill.load(in_strm)
        preds = model.predict_proba(pd.DataFrame({'comment_text': [comment]}))[:,1][0]

        return '''
                <h1>The comment is toxic with probability in:  {}</h1>'''.format((preds*100))

    # otherwise handle the GET request
    return '''
           <form method="POST"> 
                <h2>Input comment for checking</h2>           
               <div><label> <input type="text" name="comment" size="100"></label></div>
               <br>
               <div><input type="submit" value="Check" size="20"></div>
           </form>'''

if __name__ == '__main__':
    app.run(debug=True)