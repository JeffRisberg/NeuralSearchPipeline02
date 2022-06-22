import pandas as pd
import string
import ast
import json

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever import BM25Retriever, EmbeddingRetriever

from haystack.pipelines import Pipeline
from haystack.nodes.other import JoinDocuments


def clean_text(text):
  return ''.join(x for x in text.lower() if x in string.printable).replace(
      '.', '')


def load_data():
  content_dicts = []

  df = pd.read_csv('../data/carta_kbs_with_entities.csv')
  df = df.reset_index()  # make sure indexes pair with number of rows

  df.fillna("", inplace = True)

  df['parsed_content'] = df['content']
  parsed_contents = []
  for i, row in df.iterrows():
    content = ""
    linkedContent = json.loads(row['content'])
    if 'linked' in linkedContent:
      for c in linkedContent['linked']:
        if 'list' in c:
          if 'item' in c['list']:
            for element in c['list']['item']:
              if 'text' in element:
                content += '\n' + element['text']
        if 'text' in c:
          content += '\n' + c['text']['text']

    parsed_contents.append(content)
    df.at[i,'parsed_content'] = parsed_contents

  for index, row in df.iterrows():
    list_of_entities = ast.literal_eval(row['entities'])
    key = " ".join([row['title']] + [row['subject']] + list_of_entities)
    content_dict = {
      'content': clean_text(row['title'] + " " + row['subject']),
      'meta': {'content_id': row['content_id'],
               "title": row['title'],
               "subject": row['subject'],
               "row": index
               }
    }
    content_dicts.append(content_dict)
  return content_dicts

def build_pipeline():
  print("building pipeline")
  content_dicts = load_data()
  print(len(content_dicts), "articles to index")

  bm_document_store = ElasticsearchDocumentStore(similarity="cosine",
                                                 embedding_dim=384,
                                                 index="bm_documents")
  bm_document_store.delete_documents()
  bm_document_store.write_documents(content_dicts)

  bm_retriever = BM25Retriever(document_store=bm_document_store)

  es_document_store = ElasticsearchDocumentStore(similarity="cosine",
                                                 embedding_dim=384,
                                                 index="es_documents")
  es_document_store.delete_documents()
  es_document_store.write_documents(content_dicts)

  model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
  e_retriever = EmbeddingRetriever(
      document_store=es_document_store,
      embedding_model=model_name
  )

  print("begin update es_document_store embeddings")
  es_document_store.update_embeddings(e_retriever)
  print("end update es_document_store embeddings")

  combined_p = Pipeline()
  combined_p.add_node(component=bm_retriever, name="BMRetriever",
                      inputs=["Query"])

  combined_p.add_node(component=e_retriever, name="ESRetriever",
                      inputs=["Query"])

  combined_p.add_node(
    component=JoinDocuments(join_mode="merge", weights=[0.5, 0.5]),
    name="JoinResults_content",
    inputs=["BMRetriever", "ESRetriever"])

  print("pipeline complete")
  return combined_p


if __name__ == '__main__':
  pipeline = build_pipeline()

  query = "I want to speak to someone about my 409A valuation"
  result = pipeline.run(
      query=clean_text(query),
      params={
        "BMRetriever": {"top_k": 3},
        "ESRetriever": {"top_k": 3}
      })

  for r in result['documents']:
    print(r.meta['title'])
    print(r.meta['subject'])
    print(r.score)
    print()
