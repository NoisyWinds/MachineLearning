import flask
from pred import *
app = flask.Flask(__name__)
@app.route('/',methods=['GET'])
def index():
	return flask.render_template('index.html')

@app.route('/',methods=['POST'])
def predition():
	dataURL = flask.request.form.get('dataURL')
	result = pred.response(dataURL)
	return result
	
if __name__ == '__main__':
	pred = Pred()
	app.run()