from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
load_dotenv()

index_name = "school-kb"
prompt = "How can i write a single 'helloWorld' GET endpoint in express.js?"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


# returns most similar documents given the prompt
result = vectorstore.similarity_search(prompt, k=2)
# print(result)

context = " ".join([source.page_content for source in result])
# print(context)

prompt_template = ChatPromptTemplate.from_messages([
  ("system", """You are a helpful assistant that knows everything about the backend web framework express.js. 
   If you provide code examples, please be sure to always comment on a bulleted list the explanation of the code and what it is doing."""),
  # le pasamos el prompt al prompt template
  ("human", "{prompt}")
])

model = ChatOpenAI()
parser = StrOutputParser()
chain = prompt_template | model | parser

response = chain.invoke({
  "context": context,
  "prompt": prompt
})

print(response)
