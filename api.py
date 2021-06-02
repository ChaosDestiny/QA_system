import flask
from flask import request, jsonify
from getAnswer import answer

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/qa-api', methods=['GET'])
def get_answer():
    query_params = request.args
    txt = query_params.get('text')
    sim_q, ans = answer(txt)
    result = {'question': txt, 'first': {'similar question1': sim_q[0], 'answer1': ans[0]},
            'second': {'similar question2': sim_q[1], 'answer1': ans[1]},
            'third': {'similar question3': sim_q[2], 'answer1': ans[2]}}
    return jsonify(result)


if __name__ == '__main__':
    app.run()
