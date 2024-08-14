import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import sys
import logging
from colorama     import Fore
import argparse
import time


class LLM():
    def __init__(self, name=False, not_streamlit=False, list_of_questions=False):
        self.st = not not_streamlit
        if self.st:
            self.name = 'juego'
        else:
            self.name = name
        if not self.st and not self.name:
            raise Exception("Falta el nombre de la carpeta de la bd!")
        self.list_of_questions = list_of_questions
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local(self.name,self.embeddings,allow_dangerous_deserialization=True)
        
    def get_llm(self):
        #Método que recibe el nombre de la carpeta de la bd vectorial con sus index
        #y devuelve el chain de llm con el pipeline que incluye el modelo

        llm = LlamaCpp(
        streaming = True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0,
        n_ctx=4096
                )
        template = """
                    You are an assistant and you have to answer questions about the book Juego de Tronos- Canción de Hielo y Fuego 1.
                    Try to get as much information as possible in the book about the characters, the relationship between them, the events and when they were carried out.
                    With that answer the question just based on the context below and the information in the book. If you can't 
                    answer the question, reply "I don't know". Don't make up answers.
                    
                    Context: {context}
                    Question: {question}
                """
        prompt = PromptTemplate.from_template(template)
        
        parser = StrOutputParser()
        retriever = self.vector_store.as_retriever()

        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | prompt
            | llm|parser
            
        )

        return chain


    def main(self):
        

        #Ejecucion con streamlit
        if self.st:
            st.title('Chat sobre Juego de Tronos :dragon: :wolf: :european_castle: :crossed_swords: :snowflake:')
            #Inicializamos el chain
            chain = self.get_llm()
            #Chat History
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []

            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])

            #Conversación
            if message:= st.chat_input("Hazme una pregunta: "):
                if message == 'stop':
                    with st.chat_message("assistant"):
                        st.markdown("Gracias por usar el chatbot!")
                    st.stop()
                    return
                st.chat_message("user").markdown(message)
                st.session_state['chat_history'].append({'role':'user','content':message})
                print(Fore.BLUE, f"Question: {message} ", Fore.RESET)
                
                #Printeo de busqueda de similiridad de docs en la bd con el score de distancia L2
                docs = self.vector_store.similarity_search_with_score(message)
                for doc in docs:
                    print("Documentos similares:")
                    print(doc[0].page_content)
                    print(f'Distancia euclideana: {doc[1]}')
                    print('-------'*10)
                answer = chain.invoke({'question':message})
                print(Fore.RED, f"Answer: {answer} ", Fore.RESET)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                #Se agrega la respuesta al chat_history
                st.session_state.chat_history.append({"role":"assistant","content":answer})

        #Ejecucion por terminal.
        else:
            #Inicializamos el chain
            chain = self.get_llm()
            if not self.list_of_questions:
                print(Fore.RED, f"Escriba 'stop' si quiere finalizar la sesión", Fore.RESET)
                question = input("Haz una pregunta: ")
                while not question == 'stop':
                    docs = self.vector_store.similarity_search_with_score(question)
                    for doc in docs:
                        print("Documentos similares:")
                        print(doc[0].page_content)
                        print(f'Distancia euclideana: {doc[1]}')
                        print('-------'*10)
                    answer = chain.invoke({'question':question})
                    print(Fore.RED, f"Answer: {answer} ", Fore.RESET)
                    question = input("Haz una pregunta: ")
                return
            else:
                #Respondiendo desde un archivo con preguntas
                with open(self.list_of_questions, 'r') as f:
                    for question in f:
                        print(Fore.BLUE, f"Question: {question} ", Fore.RESET)
                        answer = chain.invoke({'question':question})
                        print(Fore.RED, f"Answer: {answer} ", Fore.RESET)
                return



    def get_giskard_test(self):
        #Método para evaluar el RAG con Giskard, requiere OPENAI_API_KEY ya que evalua con chatgpt4
        #genera un reporte html con el nombre report_{self.name}.html
        #FOR GISKARD TEST
        from giskard.rag import KnowledgeBase
        from giskard.rag import generate_testset
        import pandas as pd
        from giskard.rag import evaluate
        from giskard.rag import QATestset
        import os 

        OPENAI_API_KEY = ''

        def answer_fn(question):
            chain = self.get_llm()
            return chain.invoke({"question": question})
        
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        #se construye la knowledge base con los documentos generados en csv en la ingestión de la bd vectorial
        df = pd.read_csv(f'./csv/{self.name}.csv')
        knowledge_base = KnowledgeBase(df)

        #giskard genera el testset de 20 preguntas
        testset = generate_testset(
            knowledge_base,
            num_questions=20,
            agent_description="A chatbot answering questions about Harry Potter y la Piedra Filosofal",
        )

        #QATesestset pasa el listado de preguntas test a chat gpt4
        test_set_df = testset.to_pandas()
        testset = QATestset.load(test_set_df)

        #El reporte compara las respuestas del chatgpt4 de qatest vs la de nuestro modelo que actua en la funcion answer_fn
        report = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)
        report.to_html("report_{self.name}.html")


class StreamToLogger(object):
    """
        Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
            temp_linebuf = self.linebuf + buf
            self.linebuf = ''
            for line in temp_linebuf.splitlines(True):
                if line[-1] == '\n':
                    self.logger.log(self.log_level, line.rstrip())
                else:
                    self.linebuf += line

    def flush(self):
            if self.linebuf != '':
                self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ''

if __name__ =='__main__':
    
    #ARGPARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help = 'Nombre de la vectordb', type=str)
    parser.add_argument('--nst', help = 'Ejecución sin streamlit', type=bool)
    parser.add_argument('--gk', help = 'Ejecución del test con giskard', type=bool)
    parser.add_argument('--lq', help = 'Ejecución con lista de preguntas, pasar el nombre del archivo', type=str)
    args, optional_args = parser.parse_known_args()
    vectordb_name = args.n
    not_st_execution = args.nst 
    gk_test = args.gk 
    lq_test = args.lq
    
    #Giskard Test
    if gk_test:
        llm = LLM(name=vectordb_name)
        llm.giskard_test()
    #List of questions
    elif lq_test:
        llm = LLM(name=vectordb_name, not_streamlit=True, list_of_questions=  lq_test) 
        llm.main() 
    #Preguntas por terminal
    elif not_st_execution:
        llm = LLM(name=vectordb_name, not_streamlit=True) 
        llm.main() 
    #Streamlit
    else:
        #LOGHANDLER
        logger = logging.getLogger('logs')
        logger.handlers = []
        date = time.strftime("%Y-%m-%d")
        fh = logging.FileHandler(filename=f'./{date}', mode = 'a')
        formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel("DEBUG")
        wrapped_logger = StreamToLogger(logger, logging.INFO)
        sys.stdout = wrapped_logger

        #LLM CON STREAMLIT
        llm = LLM(name=vectordb_name)
        llm.main()

        #print LOG
        sys.stdout = sys.__stdout__
        with open(f"./{date}") as f:
            l = f.readlines()
            for line in l:
                print(line)

