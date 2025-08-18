# !pip install llama-index==0.12.49 \
#     llama-index-embeddings-huggingface==0.5.5 \
#     llama-index-llms-ibm==0.4.0 \
#     llama-index-retrievers-bm25==0.5.2 \
#     sentence-transformers==5.0.0 \
#     rank-bm25==0.2.2 \
#     PyStemmer==2.2.0.3 \
#     ibm-watsonx-ai==1.3.31 | tail -n 1


import os
import json
from typing import List, Optional
import asyncio
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

# IBM WatsonX LlamaIndex integration
from ibm_watsonx_ai import APIClient
from llama_index.llms.ibm import WatsonxLLM

# Sentence transformers
from sentence_transformers import SentenceTransformer

# Statistical libraries for fusion techniques
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some advanced fusion features will be limited")

print("✅ All imports successful!")

# watsonx.ai LLM using official LlamaIndex integration
def create_watsonx_llm():
    """Create watsonx.ai LLM instance using official LlamaIndex integration."""
    try:
        # Create the API client object
        api_client = APIClient({'url': "https://us-south.ml.cloud.ibm.com"})
        # Use llama-index-llms-ibm (official watsonx.ai integration)
        llm = WatsonxLLM(
            model_id="ibm/granite-3-3-8b-instruct",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            api_client=api_client,
            temperature=0.9
        )
        print("✅ watsonx.ai LLM initialized using official LlamaIndex integration")
        return llm
    except Exception as e:
        print(f"⚠️ watsonx.ai initialization error: {e}")
        print("Falling back to mock LLM for demonstration")
        
        # Fallback mock LLM for demonstration
        from llama_index.core.llms.mock import MockLLM
        return MockLLM(max_tokens=512)
    
# Initialize embedding model first
print("🔧 Initializing HuggingFace embeddings...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
print("✅ HuggingFace embeddings initialized!")

# Setup with watsonx.ai
print("🔧 Initializing watsonx.ai LLM...")
llm = create_watsonx_llm()

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model
print("✅ watsonx.ai LLM and embeddings configured!")


# Sample data for the lab - AI/ML focused documents
SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout the lab
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning", 
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}

print(f"📄 Loaded {len(SAMPLE_DOCUMENTS)} sample documents")
print(f"🔍 Prepared {len(DEMO_QUERIES)} consistent demo queries")
for i, doc in enumerate(SAMPLE_DOCUMENTS[:3], 1):
    print(f"{i}. {doc}")
print("...")

class AdvancedRetrieversLab:
    def __init__(self):
        print("🚀 Initializing Advanced Retrievers Lab...")
        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes = SentenceSplitter().get_nodes_from_documents(self.documents)
        
        print("📊 Creating indexes...")
        # Create various indexes
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)
        
        print("✅ Advanced Retrievers Lab Initialized!")
        print(f"📄 Loaded {len(self.documents)} documents")
        print(f"🔢 Created {len(self.nodes)} nodes")

# Initialize the lab
lab = AdvancedRetrieversLab()

# Vector Index Retriever

# How it works:
#     Documents are split into nodes and embedded using the configured embedding model
#     Query is converted to an embedding vector

# When to use:

#     General-purpose semantic search (most common use case)
#     Finding conceptually related content based on meaning rather than exact keywords
#     RAG pipelines where semantic understanding is crucial

print("=" * 60)
print("1. VECTOR INDEX RETRIEVER")
print("=" * 60)

# Basic vector retriever
vector_retriever = VectorIndexRetriever(
    index=lab.vector_index,
    similarity_top_k=3
)

# Alternative creation method
alt_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["basic"]  # "What is machine learning?"
nodes = vector_retriever.retrieve(query)

print(f"Query: {query}")
print(f"Retrieved {len(nodes)} nodes:")
for i, node in enumerate(nodes, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:100]}...")
    print()


# BM25 Retriever 

# Ideal for:

#     Technical documentation where exact terms matter
#     Legal documents with specific terminology
#     Product catalogs with precise specifications

# Advantages:

#     Excellent precision for exact term matches
#     Fast computational performance
#     Proven effectiveness in production systems


print("=" * 60)
print("2. BM25 RETRIEVER")
print("=" * 60)

try:
    import Stemmer
    
    # Create BM25 retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=lab.nodes,
        similarity_top_k=3,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )
    
    query = DEMO_QUERIES["technical"]  # "neural networks deep learning"
    nodes = bm25_retriever.retrieve(query)
    
    print(f"Query: {query}")
    print("BM25 analyzes exact keyword matches with sophisticated scoring")
    print(f"Retrieved {len(nodes)} nodes:")
    
    for i, node in enumerate(nodes, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0
        print(f"{i}. BM25 Score: {score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        
        # Highlight which query terms appear in the text
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        if found_terms:
            print(f"   → Found terms: {found_terms}")
        print()
    
    print("BM25 vs TF-IDF Comparison:")
    print("TF-IDF Problem: Linear term frequency scaling")
    print("  Example: 10 occurrences → score of 10, 100 occurrences → score of 100")
    print("BM25 Solution: Saturation function")
    print("  Example: 10 occurrences → high score, 100 occurrences → slightly higher score")
    print()
    print("TF-IDF Problem: No document length consideration")
    print("  Example: Long documents dominate results")
    print("BM25 Solution: Length normalization (b parameter)")
    print("  Example: Scores adjusted based on document length vs. average")
    print()
    print("Key BM25 Parameters:")
    print("- k1 ≈ 1.2: Term frequency saturation (how quickly scores plateau)")
    print("- b ≈ 0.75: Document length normalization (0=none, 1=full)")
    print("- IDF weighting: Rare terms get higher scores")
        
except ImportError:
    print("⚠️ BM25Retriever requires 'pip install PyStemmer'")
    print("Demonstrating BM25 concepts with fallback vector search...")
    
    fallback_retriever = lab.vector_index.as_retriever(similarity_top_k=3)
    query = DEMO_QUERIES["technical"]
    nodes = fallback_retriever.retrieve(query)
    
    print(f"Query: {query}")
    print("(Using vector fallback to demonstrate BM25 concepts)")
    
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Vector Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        
        # Demonstrate TF-IDF concept manually
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        
        if found_terms:
            print(f"   → BM25 would boost this result for terms: {found_terms}")
        print()
    
    print("BM25 Concept Demonstration:")
    print("1. TF-IDF Foundation:")
    print("   - Term Frequency: How often words appear in document")
    print("   - Inverse Document Frequency: How rare words are across collection")
    print("   - TF-IDF = TF × IDF (balances frequency vs rarity)")
    print()
    print("2. BM25 Improvements:")
    print("   - Saturation: Prevents over-scoring repeated terms")
    print("   - Length normalization: Prevents long document bias")
    print("   - Tunable parameters: k1 (saturation) and b (length adjustment)")
    print()
    print("3. Real-world Usage:")
    print("   - Elasticsearch default scoring function")
    print("   - Apache Lucene/Solr standard")
    print("   - Used in 83% of text-based recommender systems")
    print("   - Developed by Robertson & Spärck Jones at City University London")


# Document Summary Index Retrievers
# Document Summary Index Retrievers use document summaries instead 
# of the actual documents to find relevant content, making them efficient 
# for large collections. They return the original documents, not their summaries.


# Configuration Parameters:

#     choice_top_k (LLM retriever): Number of documents to select
#     similarity_top_k (Embedding retriever): Number of documents to select
#     Default is 1, increase for multiple document retrieval

print("=" * 60)
print("3. DOCUMENT SUMMARY INDEX RETRIEVERS")
print("=" * 60)

# LLM-based document summary retriever
doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
    lab.document_summary_index,
    choice_top_k=3  # Number of documents to select
)

# Embedding-based document summary retriever  
doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
    lab.document_summary_index,
    similarity_top_k=3  # Number of documents to select
)

query = DEMO_QUERIES["learning_types"]  # "different types of learning"

print(f"Query: {query}")

print("\nA) LLM-based Document Summary Retriever:")
print("Uses LLM to select relevant documents based on summaries")
try:
    nodes_llm = doc_summary_retriever_llm.retrieve(query)
    print(f"Retrieved {len(nodes_llm)} nodes")
    for i, node in enumerate(nodes_llm[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f"   Text: {node.text[:80]}...")
        print()
except Exception as e:
    print(f"LLM-based retrieval demo: {str(e)[:100]}...")

print("B) Embedding-based Document Summary Retriever:")
print("Uses vector similarity between query and document summaries")
try:
    nodes_emb = doc_summary_retriever_embedding.retrieve(query)
    print(f"Retrieved {len(nodes_emb)} nodes")
    for i, node in enumerate(nodes_emb[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f"   Text: {node.text[:80]}...")
        print()
except Exception as e:
    print(f"Embedding-based retrieval demo: {str(e)[:100]}...")

print("Document Summary Index workflow:")
print("1. Generates summaries for each document using LLM")
print("2. Uses summaries to select relevant documents")
print("3. Returns full content from selected documents")

# Auto Merging Retriever - Hierarchical Context Preservation

# Auto Merging Retriever is designed to preserve context in long documents using
# a hierarchical structure. It uses hierarchical chunking to break documents into 
# parent and child nodes, and if enough child nodes from 
# the same parent are retrieved, the retriever returns the parent node instead.

# Configuration:

#     chunk_sizes: List of chunk sizes from largest to smallest (e.g., [512, 256, 128])
#     chunk_overlap: Overlap between chunks to maintain continuity
#     Storage context manages both vector store (child nodes) and docstore (parent nodes)

print("=" * 60)
print("4. AUTO MERGING RETRIEVER")
print("=" * 60)

# Create hierarchical nodes
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 256, 128]
)

hier_nodes = node_parser.get_nodes_from_documents(lab.documents)

# Create storage context with all nodes
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

docstore = SimpleDocumentStore()
docstore.add_documents(hier_nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

# Create base index
base_index = VectorStoreIndex(hier_nodes, storage_context=storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=6)

# Create auto-merging retriever
auto_merging_retriever = AutoMergingRetriever(
    base_retriever, 
    storage_context,
    verbose=True
)

query = DEMO_QUERIES["advanced"]  # "How do neural networks work in deep learning?"
nodes = auto_merging_retriever.retrieve(query)

print(f"Query: {query}")
print(f"Auto-merged to {len(nodes)} nodes")
for i, node in enumerate(nodes[:3], 1):
    print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Auto-merged)")
    print(f"   Text: {node.text[:120]}...")
    print()


# Recursive Retriever - Multi-Level Reference Following

# The Recursive Retriever is designed to follow relationships between nodes 
# using references. It can follow references from one node to another, such as
# citations in academic papers or other metadata links, allowing 
# it to retrieve related content across documents or layers of abstraction.


# Configuration:

#     retriever_dict: Maps node IDs or keys to specific retrievers
#     query_engine_dict: Maps keys to query engines for sub-queries
#     Node metadata can contain references to other nodes or data structures


print("=" * 60)
print("5. RECURSIVE RETRIEVER")
print("=" * 60)

# Create documents with references
docs_with_refs = []
for i, doc in enumerate(lab.documents):
    # Add reference metadata
    ref_doc = Document(
        text=doc.text,
        metadata={
            "doc_id": f"doc_{i}",
            "references": [f"doc_{j}" for j in range(len(lab.documents)) if j != i][:2]
        }
    )
    docs_with_refs.append(ref_doc)

# Create index with referenced documents
ref_index = VectorStoreIndex.from_documents(docs_with_refs)

# Create retriever mapping
retriever_dict = {
    f"doc_{i}": ref_index.as_retriever(similarity_top_k=1)
    for i in range(len(docs_with_refs))
}

# Base retriever
base_retriever = ref_index.as_retriever(similarity_top_k=2)

# Add the root retriever to the dictionary
retriever_dict["vector"] = base_retriever

# Recursive retriever
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict=retriever_dict,
    query_engine_dict={},
    verbose=True
)

query = DEMO_QUERIES["applications"]  # "What are the applications of AI?"
try:
    nodes = recursive_retriever.retrieve(query)
    print(f"Query: {query}")
    print(f"Recursively retrieved {len(nodes)} nodes")
    for i, node in enumerate(nodes[:3], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Recursive)")
        print(f"   Text: {node.text[:100]}...")
        print()
except Exception as e:
    print(f"Query: {query}")
    print(f"Recursive retriever demo: {str(e)}")
    print("Note: Recursive retriever requires specific node reference setup")
    
    # Fallback to basic retrieval for demonstration
    print("\nFalling back to basic retrieval demonstration...")
    base_nodes = base_retriever.retrieve(query)
    for i, node in enumerate(base_nodes[:2], 1):
        print(f"{i}. Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()

# Query Fusion Retriever - Multi-Query Enhancement with Advanced Fusion¶

# The Query Fusion Retriever combines results from different retrievers 
# (such as vector-based and keyword-based methods) and optionally generates 
# multiple variations of a query using an LLM to improve coverage. 
# The results are merged using fusion strategies to improve recall.


# Configuration:

#     num_queries: Number of query variations to generate (default: 4)
#     mode: Fusion strategy ("reciprocal_rerank", "relative_score", "dist_based_score")
#     similarity_top_k: Number of results to retrieve per query
#     use_async: Enable async processing for better performance

print("=" * 60)
print("6. QUERY FUSION RETRIEVER - OVERVIEW")
print("=" * 60)

# Create base retriever
base_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"
print(f"Query: {query}")
print("QueryFusionRetriever generates multiple query variations and fuses results")
print("using one of three sophisticated fusion modes.")

print("\nOverview of Fusion Modes:")
print("1. RECIPROCAL_RERANK: Uses reciprocal rank fusion (most robust)")
print("2. RELATIVE_SCORE: Preserves score magnitudes (most interpretable)")  
print("3. DIST_BASED_SCORE: Statistical normalization (most sophisticated)")

print("\nDemonstration workflow:")
print("Each subsection below explores one fusion mode in detail with:")
print("- Theoretical explanation of the fusion method")
print("- Live demonstration using QueryFusionRetriever")
print("- Manual implementation showing the underlying mathematics")
print("- Use case recommendations and trade-offs")

print(f"\nUsing consistent test query throughout: '{query}'")
print("This allows direct comparison of how each fusion mode handles the same input.")

print("\nProceed to subsections 6.1, 6.2, and 6.3 for detailed demonstrations...")

# Reciprocal Rank Fusion (RRF) Mode
# -----------------------------------------------------

print("=" * 60)
print("6.1 RECIPROCAL RANK FUSION MODE DEMONSTRATION")
print("=" * 60)

# Create QueryFusionRetriever with RRF mode
base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

print("Testing QueryFusionRetriever with reciprocal_rerank mode:")
print("This demonstrates how RRF works within the query fusion framework")

# Use the same query for consistency across all fusion modes
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with RRF mode
    rrf_query_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )
    
    print(f"\nQuery: {query}")
    print("QueryFusionRetriever will:")
    print("1. Generate query variations using LLM")
    print("2. Retrieve results for each variation")
    print("3. Apply Reciprocal Rank Fusion")
    
    nodes = rrf_query_fusion.retrieve(query)
    
    print(f"\nRRF Query Fusion Results:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Final RRF Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
    
    print("RRF Benefits in Query Fusion Context:")
    print("- Automatically handles query variations of different quality")
    print("- No bias toward queries that return higher raw scores")
    print("- Stable performance across diverse query formulations")
    
except Exception as e:
    print(f"QueryFusionRetriever error: {e}")
    print("Demonstrating RRF concept manually with query variations...")
    
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual RRF with Query Variations:")
    all_results = {}
    
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        
        # Apply RRF scoring
        for rank, node in enumerate(nodes):
            node_id = node.node.node_id
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'rrf_score': 0,
                    'query_ranks': []
                }
            
            # Calculate RRF contribution: 1 / (rank + k)
            k = 60  # Standard RRF parameter
            rrf_contribution = 1.0 / (rank + 1 + k)
            all_results[node_id]['rrf_score'] += rrf_contribution
            all_results[node_id]['query_ranks'].append((i, rank + 1))
    
    # Sort by final RRF score
    sorted_results = sorted(
        all_results.values(), 
        key=lambda x: x['rrf_score'], 
        reverse=True
    )
    
    print(f"\nCombined RRF Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Final RRF Score: {result['rrf_score']:.4f}")
        print(f"   Query ranks: {result['query_ranks']}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("RRF Formula Demonstration:")
    print("For each document: RRF_score = Σ(1 / (rank + 60))")
    print("- Rank 1 in query: 1/(1+60) = 0.0164")
    print("- Rank 2 in query: 1/(2+60) = 0.0161")
    print("- Rank 3 in query: 1/(3+60) = 0.0159")
    print("Documents appearing in multiple queries get higher combined scores")


# Relative Score Fusion Mode
# -------------------------------------

print("=" * 60)
print("6.2 RELATIVE SCORE FUSION MODE DEMONSTRATION")
print("=" * 60)

base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

print("Testing QueryFusionRetriever with relative_score mode:")
print("This mode preserves score magnitudes while normalizing across query variations")

# Use the same query for consistency across all fusion modes
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with relative score mode
    rel_score_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="relative_score",
        use_async=False,
        verbose=False
    )
    
    print(f"\nQuery: {query}")
    print("QueryFusionRetriever with relative_score will:")
    print("1. Generate query variations")
    print("2. Normalize scores within each variation (score/max_score)")
    print("3. Combine normalized scores")
    
    nodes = rel_score_fusion.retrieve(query)
    
    print(f"\nRelative Score Fusion Results:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Combined Relative Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
    
    print("Relative Score Benefits in Query Fusion:")
    print("- Preserves confidence information from embedding model")
    print("- Ensures fair contribution from each query variation")
    print("- More interpretable than rank-only methods")
    
except Exception as e:
    print(f"QueryFusionRetriever error: {e}")
    print("Demonstrating Relative Score concept manually...")
    
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual Relative Score Fusion with Query Variations:")
    all_results = {}
    query_max_scores = []
    
    # Step 1: Get results and find max scores for each query
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        max_score = max(scores) if scores else 1.0
        query_max_scores.append(max_score)
        
        print(f"Max score for this query: {max_score:.4f}")
        
        # Store results with normalization info
        for node in nodes:
            node_id = node.node.node_id
            original_score = node.score or 0
            normalized_score = original_score / max_score if max_score > 0 else 0
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': original_score,
                'normalized': normalized_score
            })
    
    # Step 2: Sort by combined relative score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    print(f"\nCombined Relative Score Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Combined Score: {result['combined_score']:.4f}")
        print(f"   Score breakdown:")
        for contrib in result['contributions']:
            print(f"     Query {contrib['query']}: {contrib['original']:.3f} → {contrib['normalized']:.3f}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("Relative Score Normalization Process:")
    print("1. For each query variation, find max_score")
    print("2. Normalize: normalized_score = original_score / max_score")
    print("3. Sum normalized scores across all query variations")
    print("4. Documents with consistently high scores across queries win")


# Distribution-Based Score Fusion Mode
# -----------------------------------------------------


print("=" * 60)
print("6.3 DISTRIBUTION-BASED SCORE FUSION MODE DEMONSTRATION")
print("=" * 60)

base_retriever = lab.vector_index.as_retriever(similarity_top_k=8)

print("Testing QueryFusionRetriever with dist_based_score mode:")
print("This mode uses statistical analysis for the most sophisticated score fusion")

# Use the same query for consistency across all fusion modes
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with distribution-based mode
    dist_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="dist_based_score",
        use_async=False,
        verbose=False
    )
    
    print(f"\nQuery: {query}")
    print("QueryFusionRetriever with dist_based_score will:")
    print("1. Generate query variations")
    print("2. Analyze score distributions for each variation")
    print("3. Apply statistical normalization (z-score, percentiles)")
    print("4. Combine with distribution-aware weighting")
    
    nodes = dist_fusion.retrieve(query)
    
    print(f"\nDistribution-Based Fusion Results:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Statistically Normalized Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
    
    print("Distribution-Based Benefits in Query Fusion:")
    print("- Accounts for score distribution differences between query variations")
    print("- Statistically robust against outliers and noise")
    print("- Adapts weighting based on query variation reliability")
    
except Exception as e:
    print(f"QueryFusionRetriever error: {e}")
    print("Demonstrating Distribution-Based concept manually...")
    
    if not SCIPY_AVAILABLE:
        print("⚠️ Full statistical analysis requires scipy")
    
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual Distribution-Based Fusion with Query Variations:")
    all_results = {}
    variation_stats = []
    
    # Step 1: Collect results and analyze distributions
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        
        # Calculate distribution statistics
        mean_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if len(scores) > 1 else 1
        min_score = np.min(scores) if scores else 0
        max_score = np.max(scores) if scores else 1
        
        stats_info = {
            'mean': mean_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'nodes': nodes,
            'scores': scores
        }
        variation_stats.append(stats_info)
        
        print(f"Distribution stats: mean={mean_score:.3f}, std={std_score:.3f}")
        print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")
        
        # Apply z-score normalization
        for node, score in zip(nodes, scores):
            node_id = node.node.node_id
            
            # Z-score normalization
            if std_score > 0:
                z_score = (score - mean_score) / std_score
            else:
                z_score = 0
            
            # Convert to [0,1] using sigmoid
            normalized_score = 1 / (1 + np.exp(-z_score))
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': score,
                'z_score': z_score,
                'normalized': normalized_score
            })
    
    # Step 2: Sort by combined distribution-based score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    print(f"\nCombined Distribution-Based Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Combined Score: {result['combined_score']:.4f}")
        print(f"   Statistical breakdown:")
        for contrib in result['contributions']:
            print(f"     Query {contrib['query']}: {contrib['original']:.3f} → "
                  f"z={contrib['z_score']:.2f} → {contrib['normalized']:.3f}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("Distribution-Based Process:")
    print("1. Calculate mean and std for each query variation")
    print("2. Z-score normalize: z = (score - mean) / std")
    print("3. Sigmoid transform: normalized = 1 / (1 + exp(-z))")
    print("4. Sum normalized scores across variations")
    print("5. Results reflect statistical significance across all query forms")

# Show fusion mode comparison summary
print("\n" + "=" * 60)
print("FUSION MODES COMPARISON SUMMARY")
print("=" * 60)
print("All three modes tested with the same query for direct comparison:")
print(f"Query: {query}")
print()
print("Mode Characteristics:")
print("• RRF (reciprocal_rerank): Most robust, rank-based, scale-invariant")
print("• Relative Score: Preserves confidence, normalizes by max score")  
print("• Distribution-Based: Most sophisticated, statistical normalization")
print()
print("Choose based on your use case:")
print("- Production stability → RRF")
print("- Score interpretability → Relative Score")
print("- Statistical robustness → Distribution-Based")



# Recommended Retrievers by Use Case

# Based on the authoritative source and the characteristics of each retriever, here are recommended approaches for different scenarios:

# General Q&A Applications:

#     Primary: Vector Index Retriever for semantic understanding
#     Enhancement: Combine with BM25 Retriever using Query Fusion for hybrid approach
#     Benefit: Combines semantic relevance with keyword matching
#     From authoritative source: "For general Q&A, use a vector index retriever, potentially combined with a BM25 retriever. This retriever fusion combines semantic relevance with keyword matching."

# Technical Documentation:

#     Primary: BM25 Retriever for exact term matching
#     Enhancement: Vector Index Retriever as secondary for contextual flexibility
#     Benefit: Prioritizes exact technical terms while maintaining semantic understanding
#     From authoritative source: "For technical documents, especially those where exact terms need to be prioritized, consider making BM25 your primary retriever, with the vector index retriever adding contextual flexibility as a secondary retriever."

# Long Documents:

#     Primary: Auto Merging Retriever
#     Benefit: Retrieves longer parent versions only if enough shorter child versions are retrieved, preserving context
#     From authoritative source: "For long documents, the auto merging retriever is a great option, because it will retrieve longer parent versions only if enough shorter child versions are retrieved."

# Research Papers:

#     Primary: Recursive Retriever
#     Benefit: Follows citations and references to retrieve relevant content from cited papers
#     From authoritative source: "For research papers, use the recursive retriever in order to retrieve relevant content from cited papers."

# Large Document Collections:

#     Primary: Document Summary Index Retriever for initial filtering
#     Enhancement: Followed by Vector Index Retriever for detailed search within relevant documents
#     Benefit: Narrows down relevant documents first, then performs detailed retrieval
#     From authoritative source: "For large document sets, consider using the document summary index retriever to narrow down the number of relevant documents, followed by a vector search within the remaining subset to retrieve the most pertinent content."
