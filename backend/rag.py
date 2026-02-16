from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def get_qa_chain():

    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="qwen/qwen3-32b",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant.
        Answer ONLY using the provided context.
        If the answer is not in the context, say:
        'I could not find that information on the website.'"""),
                ("human",
                """Context:
        {context}

        Question:
        {input}""")
    ])

    # Step 1: document → answer chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Step 2: retrieval → document chain
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain
