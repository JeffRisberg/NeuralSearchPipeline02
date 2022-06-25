import string
from pathlib import Path

from haystack import Pipeline
from haystack.pipelines.config import validate_yaml

def clean_text(text):
  return ''.join(x for x in text.lower() if x in string.printable).replace(
      '.', '')

validate_yaml(Path("sample.haystack-pipeline.yml"))

def show_results(results):
  print(len(results['documents']), "documents")
  for document in results['documents']:
    print(document.meta)
    print(document.score)

def run_query(query):
  print("query:", query)
  results = pipeline.run(query=clean_text(query))
  show_results(results)
  print()

pipeline = Pipeline.load_from_yaml(Path("sample.haystack-pipeline.yml"), pipeline_name='my_query_pipeline')

query = "Convertible Notes and SAFE Terms and Definitions"
run_query(query)

query = "Can I contact someone to discuss my 409A valuation?"
run_query(query)

