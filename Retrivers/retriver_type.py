from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM

from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader

from langchain.retrievers.multi_query import MultiQueryRetriever

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def llm():
    model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'
    
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    
    project_id = "skills-network"
    
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    mixtral_llm = WatsonxLLM(model = model)
    return mixtral_llm


def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks



def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

#Vector Store-Backed Retriever
loader = TextLoader("companypolicies.txt")
txt_data = loader.load()
chunks_txt = text_splitter(txt_data, 200, 20)
vectordb = Chroma.from_documents(chunks_txt, watsonx_embedding())
query = "email policy"
retriever = vectordb.as_retriever()
docs = retriever.invoke(query)

# specify search kwargs like k to limit the retrieval results.
retriever = vectordb.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke(query)
docs


#MMR search- It selects documents that are both highly relevant to the query 
#           and minimally similar to previously selected documents. This 
#           approach helps to avoid redundancy and ensures a more 
#           comprehensive coverage of different aspects of the query.

retriever = vectordb.as_retriever(search_type="mmr")
docs = retriever.invoke(query)
docs


# Multi-Query Retriever
loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf")
pdf_data = loader.load()

# Split
chunks_pdf = text_splitter(pdf_data, 500, 20)

# VectorDB
ids = vectordb.get()["ids"]
vectordb.delete(ids) # We need to delete existing embeddings from previous documents and then store current document embeddings in.
vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())


query = "What does the paper say about langchain?"

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm()
)
docs = retriever.invoke(query)
docs

# Self-Querying Retriever -

from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from lark import lark


docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]


vectordb = Chroma.from_documents(docs, watsonx_embedding())

document_content_description = "Brief summary of a movie."

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)

# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")

# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")


# Parent Document Retriever

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore

# Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
parent_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='\n')

vectordb = Chroma(
    collection_name="split_parents", embedding_function=watsonx_embedding()
)
#vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
# The storage layer for the parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(txt_data)

len(list(store.yield_keys()))
#retrieves the small chunks.
sub_docs = vectordb.similarity_search("smoking policy")
print(sub_docs[0].page_content)
#retrieve the relevant large chunk.
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)

