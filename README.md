# Chat_RAG

## Steps to Run the Project

Follow these steps to set up and run the Chat_RAG project:

1. **Clone the Repository:**
    Begin by cloning the repository to your local machine using the following command:
    ```sh
    git clone https://github.com/Samilincoln/Chat_RAG.git
    ```

2. **Navigate to the Project Directory:**
    Change your current directory to the project directory:
    ```sh
    cd Chat_RAG
    ```

3. **Install Required Dependencies:**
    Install all the necessary dependencies specified in the `requirements.txt` file:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**
    - Create a `.env` file in the root directory of the project.
    - Add your Groq API key to the `.env` file by including the following line:
      ```
      GROQ_API_KEY=your_api_key_here
      ```

5. **Navigate to the Client Directory:**
    Change your directory to the client directory where the Streamlit application is located:
    ```sh
    cd client
    ```

6. **Run the Streamlit Application:**
    Launch the Streamlit application using the following command:
    ```sh
    streamlit run app.py
    ```

By following these steps, you will have the Chat_RAG project up and running on your local machine.