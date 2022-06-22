import pandas as pd
import string
import json


def clean_text(text):
  return ''.join(x for x in text.lower() if x in string.printable).replace(
      '.', '')


def build_pipeline():
  print("building pipeline")
  kbs = pd.read_csv('../data/carta_kbs_with_entities.csv')

  kbs.fillna("", inplace = True)

  def parse_content(kb_doc, endpoint_type='KB'):

    kb_doc['parsed_content'] = kb_doc['content']
    parsed_contents = []
    for i, row in kb_doc.iterrows():
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
      kb_doc.at[i,'parsed_content'] = parsed_contents
    return kb_doc

  kbs = parse_content(kbs)

  def clean_text(text):
    return ''.join(x for x in text.lower() if x in string.printable).replace('.', '')

  deduped_dicts = []
  for i, text in kbs.iterrows():
    title_cleaned = clean_text(text['title'])
    subject_cleaned = clean_text(text['subject'])
    deduped_dicts.append({
      "title_cleaned":title_cleaned,
      "subject_cleaned":subject_cleaned,
      "title":text['title'],
      "subject":text['subject'],
      "content":text['parsed_content'],
      "entities":eval(text['entities']),
      "source":text['source'],
      "entities_category":eval(text['entities_category'])
    })

  dicts = []
  tf_dicts = []
  content_dicts = []

  for i, text in kbs.iterrows():
    dicts.append({"content": deduped_dicts[i]['title_cleaned'],
                  # ' \n '.join([
                  #     "important entity is " + ', '.join(eval(text['entities'])),
                  #     deduped_dicts[i]['title_cleaned']
                  #     ]),
                  "meta":{
                    "name":i,
                    "title":text['title'],
                    "subject":text['subject'],
                    "entities":eval(text['entities']),
                    "source":text['source'],
                    "entities_category":eval(text['entities_category']),
                    "content_id":i
                  }
                  })

    dicts.append({"content":deduped_dicts[i]['subject_cleaned'],
                  "meta":{
                    "name":i,
                    "title":text['title'],
                    "subject":text['subject'],
                    "entities":eval(text['entities']),
                    "source":text['source'],
                    "entities_category":eval(text['entities_category']),
                    #   "document_id":str(text['document_id']),
                    "content_id":i
                    #   "content_id":str(text['content_id'])
                  }
                  })

    content_dicts.append({"content":
      ' \n '.join([
        ', '.join(eval(text['entities'])),
        deduped_dicts[i]['title_cleaned'],
        deduped_dicts[i]['subject_cleaned'],
        text['content']
      ]),
      "meta":{
        "name":i,
        "title":text['title'],
        "subject":text['subject'],
        "entities":eval(text['entities']),
        "source":text['source'],
        "entities_category":eval(text['entities_category']),
        "content_id":i
      }
    })

  entity_to_dict = {}
  for dic in dicts:
    for entity in dic['meta']['entities']:
      if entity not in entity_to_dict:
        entity_to_dict[entity] = [dic['meta']['content_id']]
      else:
        entity_to_dict[entity].append(dic['meta']['content_id'])

  # ## TF Retriever for content

  from haystack.document_stores import InMemoryDocumentStore
  from haystack.nodes import TfidfRetriever
  from haystack.pipelines import ExtractiveQAPipeline

  tf_document_store = InMemoryDocumentStore()
  tf_document_store.delete_documents()
  tf_document_store.write_documents(content_dicts)

  tfretriever = TfidfRetriever(document_store=tf_document_store, auto_fit=True)

  # ## BM retriever for content

  from haystack.document_stores import ElasticsearchDocumentStore
  from haystack.nodes.retriever import EmbeddingRetriever

  bm_document_store = ElasticsearchDocumentStore(similarity="cosine",
                                                 embedding_dim=384, index="bm_documents")
  bm_document_store.delete_documents()
  bm_document_store.write_documents(content_dicts)

  from haystack.nodes import BM25Retriever

  bm_retriever = BM25Retriever(bm_document_store)

  # ## Embedding retrievers for title and subject

  e_document_store = ElasticsearchDocumentStore(similarity="cosine", embedding_dim=384, index="document")
  e_document_store.delete_documents()
  e_document_store.write_documents(dicts)

  bm_retriever_phrase = BM25Retriever(e_document_store)

  # model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
  model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'

  e_retriever = EmbeddingRetriever(
      document_store=e_document_store,
      embedding_model=model_name
  )

  e_document_store.update_embeddings(e_retriever)

  # ## RTR retriever for content

  rtr_document_store = ElasticsearchDocumentStore(similarity="dot_product", embedding_dim=128, index="rtr_documents")

  rtr_document_store.delete_documents()
  rtr_document_store.write_documents(content_dicts)

  from haystack.nodes.retriever.dense import EmbeddingRetriever

  rtr_retriever = EmbeddingRetriever(document_store=rtr_document_store,
                                     embedding_model="yjernite/retribert-base-uncased",
                                     model_format="retribert")

  rtr_document_store.update_embeddings(rtr_retriever)

  # ## QnA reader

  from haystack import Pipeline
  from haystack.nodes import JoinDocuments

  combined_p = Pipeline()
  combined_p.add_node(component=bm_retriever, name="BMRetriever", inputs=["Query"])
  combined_p.add_node(component=bm_retriever_phrase, name="BMRetriever_phrase", inputs=["Query"])
  combined_p.add_node(component=e_retriever, name="ESRetriever", inputs=["Query"])
  combined_p.add_node(component=rtr_retriever, name="RTRetriever", inputs=["Query"])

  combined_p.add_node(component = JoinDocuments(join_mode="merge", weights = [0.5, 0.5]),
                      name="JoinResults_content", inputs=[
      "BMRetriever",
      "RTRetriever"
    ])

  combined_p.add_node(component = JoinDocuments(join_mode="merge", weights = [0.5, 0.5]),
                      name="JoinResults_phrase", inputs=[
      "BMRetriever_phrase",
      "ESRetriever",
    ])

  combined_p.add_node(component = JoinDocuments(join_mode="merge"),
                      name="JoinResults", inputs=[
      "JoinResults_phrase",
      "JoinResults_content"
    ])
  print("pipeline complete")
  return combined_p


if __name__ == '__main__':
  pipeline = build_pipeline()

  inputs = ["transfer shares",
         "Cancel an Option Grant?",
         "Convertible Notes and SAFE Terms and Definitions",
         "Can I see what was submitted in the 409A request form?",
         "federal exemption question",
         "terminate option holders",
         "can i contact someone to discuss my 409a valuation?",
         "when does Carta send investor requests to the designated CEO",
         "how do I approve an Option Exercise",
         "Email preferences for Investment Firms",
         "error accepting grant"
         ]

  for input in inputs:
    print("Input phrase:", input)
    result = pipeline.run(
        query=clean_text(input),
        params={
          "BMRetriever": {"top_k": 3},
          "ESRetriever": {"top_k": 3}
        })

    for r in result['documents']:
      print(r.meta['title'])
      print(r.meta['subject'])
      print(r.score)
      print()
