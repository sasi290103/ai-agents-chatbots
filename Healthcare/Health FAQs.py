# Databricks notebook source
# Install Google Generative AI SDK
%pip install google-generativeai

# COMMAND ----------

import google.generativeai as genai

API_KEY = "Your API_Key"
genai.configure(api_key=API_KEY)

# COMMAND ----------

# DBTITLE 1,Final Code
import google.generativeai as genai
import os


API_KEY = "AIzaSyBg_0TJ_miX2UHYFjxNp9nH7EYGi9LiOJA"
genai.configure(api_key=API_KEY)

class MedicalChatbot:
    """
    A chatbot designed to answer health-related questions in a simple and friendly manner.
    It maintains a short history of the conversation to provide context.
    """
    def __init__(self, model_name="gemma-3-27b-it"):
        """
        Initializes the chatbot with a specific generative model.
        """
        try:
            self.model = genai.GenerativeModel(model_name)
            self.history = []
            print(f"Successfully initialized model: {model_name}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.model = None
            self.history = []


    def run(self, query):
        """
        Takes a user's query, adds it to the conversation history,
        generates a response from the model, and returns the answer.
        """
        if not self.model:
            return "⚠️ Error: The generative model is not initialized. Please check your API key and model name."

        # Append the user's query to the history
        self.history.append(f"Patient: {query}")
        # Use the last 5 interactions to provide context
        context = "\n".join(self.history[-5:])

        # Construct the prompt with clear instructions for the model
        prompt = f"""
        You are a medical assistant chatbot. Respond to the patient’s health-related questions in a simple, friendly, and medically accurate manner.
        Respond with clear, concise, and helpful information. Always provide information in a patient-friendly tone.
        If the query relates to symptoms, diseases, or treatments, provide accurate medical insights.
        Keep the conversation going.
        Give a short and concise answer.

        Conversation so far:
        {context}

        Chatbot:
        """

        try:
            # Generate content using the model
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            # Append the chatbot's answer to the history
            self.history.append(f"Chatbot: {answer}")
            return answer
        except Exception as e:
            # Handle potential API errors or other exceptions
            return f"⚠️ Error generating response: {str(e)}"

# --- Main Execution: Health FAQ Chatbot Interface ---
def health_faq_chatbot():
    """
    The main function to run the command-line interface for the chatbot.
    """
    # Instantiate the chatbot
    bot = MedicalChatbot()

    # Initial greeting
    print("\n--- Health FAQ Chatbot ---")
    print("Hello! I'm here to answer your health questions.")
    print("Type 'quit' or 'exit' to end the chat.")
    print("-" * 28)

    # Main loop to keep the chat going
    while True:
        # Get input from the user
        user_input = input("You: ")

        # Check if the user wants to quit
        if user_input.lower() in ["quit", "exit"]:
            print("\nChatbot: Goodbye! Take care and stay healthy.")
            break

        # Get the response from the chatbot and print it
        if user_input:
            response = bot.run(user_input)
            print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    # Run the chatbot application
    health_faq_chatbot()


# COMMAND ----------

# DBTITLE 1,MLFlow
import google.generativeai as genai
import mlflow.pyfunc

class ChatbotWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.chatbot = MedicalChatbot()
        self.chatbot.load_model()

    def predict(self, context, model_input):
        question = model_input.get("question", "")
        return {"response": self.chatbot.run(question)}

# COMMAND ----------

# DBTITLE 1,Register Model
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature

def register_chatbot_model():
    input_df = pd.DataFrame([{"question": "What are the symptoms of flu?"}])
    output_df = pd.DataFrame([{"response": "Fever, cough, sore throat, body aches, and fatigue."}])
    signature = infer_signature(input_df, output_df)

    registered_model_name = "MedicalChatbotModel"
    mlflow.set_experiment("/Users/sasidhar.tech@outlook.com/Medical_Chatbot_Registration")

    with mlflow.start_run() as run:
        print(f"\n--- Starting MLflow run: {run.info.run_id} ---")
        print(f"Registering model as: '{registered_model_name}'")

        mlflow.pyfunc.log_model(
            artifact_path="chatbot_model",
            python_model=ChatbotWrapper(),
            registered_model_name=registered_model_name, 
            pip_requirements=[                            
                "mlflow",
                "google-generativeai",
                "pandas",
                "cloudpickle==2.2.1"
            ],
            input_example=input_df,
            signature=signature
        )

        print("\n✅ Model registration complete!")
        print(f"➡️ Check MLflow UI under 'Models' tab for '{registered_model_name}'.")

register_chatbot_model()


# COMMAND ----------

# DBTITLE 1,Local Testing
# Cell / test_script.py


# 1️⃣ Test the raw MedicalChatbot logic
print("=== Direct MedicalChatbot ===")
bot = MedicalChatbot()
question = "What are the common symptoms of diabetes?"
answer = bot.run(question)
print(f"Q: {question}\nA: {answer}\n")

# 2️⃣ Test the MLflow‑style wrapper
print("=== ChatbotWrapper via pyfunc ===")
wrapper = ChatbotWrapper()
wrapper.load_context(None)  # Initializes internal MedicalChatbot
response_dict = wrapper.predict(None, {"question": question})
print(f"Q: {question}\nA: {response_dict['response']}")
