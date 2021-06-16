import flask
from flask import request, jsonify
from getAnswer import answer

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/qa-api', methods=['GET'])
def get_answer():
    query_params = request.args
    txt = query_params.get('text')
    num_id = 4
    sim_q, ids, scores = answer(txt, num_id)
    result = {'question': txt, 'result': {}}
    result['result']['sim_ques'] = []
    result['result']['id'] = []
    result['result']['score'] = []
    for i in range(len(sim_q)):
        result['result']['sim_ques'].append(sim_q[i])
        result['result']['id'].append(ids[i])
        result['result']['score'].append(scores[i])
    return jsonify(result)


if __name__ == '__main__':
    app.run()
