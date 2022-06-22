import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from prepare_haystack_4_pipeline import build_pipeline, clean_text

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Define the app
app = Flask(__name__)
# Load configs
app.config.from_object('config')
# Set CORS policies
CORS(app)

pipeline = build_pipeline()


def get_results(input):
  top_k = app.config['TOP_K']
  records = pipeline.run(
      query = clean_text(input),
      params = {
        "BMRetriever": {"top_k": top_k},
        "ESRetriever": {"top_k": top_k}
      })
  return records


@app.route("/query", methods=["GET"])
def qa():
  records = {'documents': []}

  if request.args.get("query"):
    query = request.args.get("query")

    records = pipeline.run(
        query = clean_text(query),
        params = {
          "BMRetriever": {"top_k": 3},
          "ESRetriever": {"top_k": 3}
        })
  else:
    return {"error": "Couldn't process your request"}, 422

  result = [{
    'title': r.meta['title'],
    'subject': r.meta['subject'],
    'score': r.score}
    for r in records['documents']]
  return jsonify(result)

if __name__ == '__main__':
  app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)

