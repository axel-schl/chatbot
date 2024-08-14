# PDF Chatbot con Mistral, langchain y Faiss.

## Overview
Un chatbot diseñado para responder preguntas acerca de libros de ficción en PDF.

## Technologies used
* [Mistral 7b v.01](https://huggingface.co/mistralai/Mistral-7B-v0.1): LLM Model.
* [Faiss](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/): A library for efficient similarity search and clustering of dense vectors with embeddings.
* [Streamlit](https://streamlit.io/)): Interfaz gráfica.
* [Giskard](https://www.giskard.ai/): Plataforma para testear llm con chatbot gpt 4


## Installation
* If you want to run your own build, you must have python globally installed in your computer. If you don't, you can get python [here](https://www.python.org").
* After this, confirm that you have installed virtualenv globally as well. If another case, run:
    ```bash
        $ pip install virtualenv
    ```
* Then, Git clone the repository to your PC
    ```bash
        $ git clone https://github.com/axel-schl/chatbot.git
    ```

* #### Dependencies
    1. Cd the cloned repo as such:
        ```bash
            $ cd chatbot
        ```
    2. Create and activate your virtual environment:
        ```bash
            $ virtualenv  venv -p python3
            $ source venv/bin/activate
        ```
    3. Install the requeriments needed to run the app:
        ```bash
            $ pip install -r requirements.txt
        ```
  

* #### Run the server and Sign Up

    En el directorio se encuentra el libro con el que se creo este proyecto: Juego de Tronos-Canción de Hielo y Fuego 1 (con el nombre juego.pdf).
  Se puede ejecutar el chatbot con otro libro, siempre y cuando se suba al mismo directorio, teniendo en cuenta que el prompt del llm esta especificamente diseñado
  para el libro en cuestión (solo basta con modificar el prompt con el nombre del libro que se quiere investigar con el chatbot en la variable prompt de chatbot.py)

  
    
 * #### Crear base de datos vectorial con faiss
  
  python build_vectordb.py --n book_name --cs chunk_size --co chunk_overlap

  book_name: nombre del archivo pdf dentro del directorio, en este caso 'juego' (obligatorio)
  chunk_size: tamaño de tokens nltk por chunk (opcional, por defecto 500)
  chunk_overlap: tamaño de tokens nltk de overlap (residuos) (opcional, por defecto 25)

  Se creara una carpeta index con los archivos .index y .pkl de faiss con el nombre del archivo, en este caso 'juego'.
    
 * #### Ejecutar el chatbot
    
    ##### Con streamlit:
   streamlit run chatbot.py
   Se abrirá la consola de streamlit, escribir 'stop' para cerrar la sesión.
   Tener en cuenta que para streamlit el nombre del libro ya esta predefinido en el script chatbot.py, de querer subir otro libro que 'juego'
   modificar la variable self.name en linea 19 con el nombre del libro y, por ende, de la carpeta con los archivos faiss.

   ##### Por terminal:
   python chatbot.py --nst True --n book_name
   Preguntar por terminal, escribir 'stop' para cerrar la sesión.

   ##### Con archivo listado de preguntas:
   python chatbot.py --nst True --n book_name --lq file_path

   file_path: nombre del archivo con las preguntas en el directorio (preguntas.txt es el ejemplo subido en este repo).

   
  
   * #### Testear con giskard.
    python chatbot.py --gk True
  
    Se ejecutara el test con giskard, consumiendo el archivo csv previamente creado con build_vectordb.py
   Generara un reporte en html como el subido de ejemplo (reporte.html) y un listado de preguntas generado por giskard (test-set.jsonl de ejemplo),
   que seran las preguntas que deberá responder el modelo y compararse vs. chat bot gpt 4
   Para esto se necesita setear la OPEANAI_API_KEY (linea 147)
    
