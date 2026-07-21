import os
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    """
    RAG Engine for DataLens AI using LangChain, OpenAI, and ChromaDB.
    Handles vector indexing, context retrieval, and structured QA generation with source citations.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.vector_store = None
        self.retriever = None

        if not api_key:
            raise ValueError("OpenAI API Key is required to initialize RAG Engine.")

        # Set environment variable for LangChain / OpenAI
        os.environ["OPENAI_API_KEY"] = api_key

        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.1,
            openai_api_key=api_key
        )

    def index_documents(self, documents: List[Document]) -> int:
        """Indexes dataset knowledge documents into an in-memory Chroma vector database."""
        if not documents:
            raise ValueError("No documents provided for indexing.")

        # Create Chroma vectorstore in-memory
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # Create retriever (top 4 relevant document chunks)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return len(documents)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes RAG retrieval and LLM synthesis.
        Returns a dict containing 'answer', 'sources', and 'retrieved_docs'.
        """
        if not self.retriever:
            return {
                "answer": "DataLens AI error: Vector store is not initialized. Please upload a dataset first.",
                "sources": [],
                "retrieved_docs": []
            }

        # 1. Retrieve relevant contexts
        retrieved_docs = self.retriever.get_relevant_documents(question)

        # 2. Format context string and extract sources
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            src_name = doc.metadata.get("source", f"Context Chunk #{i}")
            if src_name not in sources:
                sources.append(src_name)
            context_parts.append(f"--- Source: {src_name} ---\n{doc.page_content}")

        formatted_context = "\n\n".join(context_parts)

        # 3. RAG Prompt Template as defined in Project Overview
        prompt_template = """You are DataLens AI, an AI assistant that helps users understand and explore datasets.

Answer questions based only on the provided dataset context.

If the answer cannot be determined from the context, clearly state that the information is not available.

Do not invent statistics or column information.

When possible:
- Explain the reasoning
- Mention relevant columns
- Include relevant statistics
- Reference the source context

Dataset Context:
{context}

User Question:
{question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = prompt | self.llm | StrOutputParser()

        # 4. Generate answer
        answer = chain.invoke({
            "context": formatted_context,
            "question": question
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "retrieved_docs": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content
                }
                for doc in retrieved_docs
            ]
        }
