from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

def get_context(query, k=2):
    index_name = "school-kb"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    result = vectorstore.similarity_search(query, k=k)
    return "\n".join([source.page_content for source in result])

def create_chain():
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant. Use the following context to answer the user's question: {context}"),
        HumanMessage(content="{question}")
    ])
    model = ChatOpenAI(temperature=0.7)
    parser = StrOutputParser()
    return prompt_template | model | parser

def main():
    chain = create_chain()
    
    while True:
        user_input = input("Enter your question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        context = get_context(user_input)
        response = chain.invoke({
            "context": context,
            "question": user_input
        })
        
        print("\nAI Response:")
        print(response)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()