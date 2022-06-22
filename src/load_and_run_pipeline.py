import string
from pathlib import Path

from haystack import Pipeline
from haystack.pipelines.config import validate_yaml

def clean_text(text):
  return ''.join(x for x in text.lower() if x in string.printable).replace(
      '.', '')


pipeline = Pipeline()

validate_yaml(Path("sample.haystack-pipeline.yml"))

pipeline.load_from_yaml(Path("sample.haystack-pipeline.yml"), pipeline_name='my_query_pipeline')
print(pipeline)
print(pipeline.root_node)

result = pipeline.run(query=clean_text("Convertible Notes and SAFE Terms and Definitions"))
print(result)
