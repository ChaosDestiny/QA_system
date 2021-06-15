import flask
from flask import request, jsonify
from getAnswer import answer

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/qa-api', methods=['GET'])
def get_answer():
    query_params = request.args
    txt = query_params.get('text')
    num_id = 1
    sim_q, ids = answer(txt, num_id)
    result = {'question': txt, 'result': {}}
    for i in range(len(sim_q)):
        result['result']['sim_ques'] = sim_q[i]
        result['result']['id'] = ids[i]
    return jsonify(result)


if __name__ == '__main__':
    app.run()
