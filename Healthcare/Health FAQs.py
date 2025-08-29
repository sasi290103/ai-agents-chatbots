# Databricks notebook source
# Install Google Generative AI SDK
%pip install google-generativeai

# COMMAND ----------

import google.generativeai as genai

API_KEY = "AIzaSyBg_0TJ_miX2UHYFjxNp9nH7EYGi9LiOJA"  # Replace with actual key or pull from dbutils.secrets
genai.configure(api_key=API_KEY)

# COMMAND ----------

# DBTITLE 1,Final Code
class MedicalChatbot:
    def __init__(self, model):
        self.model = model
        self.history = []

    def run(self, query):
        self.history.append(f"Patient: {query}")
        context = "\n".join(self.history[-5:])  # Last 5 messages

        prompt = f"""
        You are a medical assistant chatbot. Respond to the patient’s health-related questions in a simple, friendly, and medically accurate manner.
        Respond with clear, concise, and helpful information. Always provide information in a patient-friendly tone.
        If the query relates to symptoms, diseases, or treatments, provide accurate medical insights.
        Keep the conversation going.
        Give a short and concise answer.

        Question: {query}

        Conversation so far:
        {context}

        Chatbot:
        """

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            self.history.append(f"Chatbot: {answer}")
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"


# COMMAND ----------

# Load the model (you might want to use "gemini-pro" instead of "gemma-3-27b-it")
model = genai.GenerativeModel("gemma-3-27b-it")  # Or "gemma-7b-it" if supported
bot = MedicalChatbot(model)

# COMMAND ----------

# Simulate chat interaction using a loop (or use dbutils.widgets for UI)
while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("🤖 Chatbot: Goodbye! Stay healthy. 🩺")
        break

    reply = bot.run(user_input)
    print(f"🤖 Chatbot: {reply}\n")

# COMMAND ----------

# DBTITLE 1,MLFlow
import google.generativeai as genai
import mlflow.pyfunc
class MedicalChatbot:
    def __init__(self):
        self.model = None
        self.history = []
    def load_model(self):
        genai.configure(api_key="AIzaSyBg_0TJ_miX2UHYFjxNp9nH7EYGi9LiOJA")
        self.model = genai.GenerativeModel("gemma-3-27b-it")
    def run(self, question: str) -> str:
        if self.model is None:
            self.load_model()

        self.history.append(f"Patient: {question}")
        context = "\n".join(self.history[-5:])
        prompt = f"""
        You are a medical assistant chatbot. Respond to the patient’s health-related questions in a simple, friendly, and medically accurate manner.
        Respond with clear, concise, and helpful information. Always provide information in a patient-friendly tone.
        If the query relates to symptoms, diseases, or treatments, provide accurate medical insights.
        Keep the conversation going.
        Give a short and concise answer.

        Question: {question}
        Conversation so far:
        {context}
        Chatbot:
        """
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            self.history.append(f"Chatbot: {answer}")
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

# ✅ MLflow wrapper
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
from mlflow.models.signature import infer_signature
import pandas as pd
from datetime import datetime, timedelta
import pytz

# 🧾 Example input
input_df = pd.DataFrame([{"question": "What are the symptoms of flu?"}])
signature = infer_signature(input_df)

timestamp_str = datetime.now(pytz.utc).strftime("%Y%m%dT%H%M%SZ")
experiment_name=f"/Workspace/AI Chabot & Agents/Experiment/{timestamp_str}"
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="chatbot_model",
        python_model=ChatbotWrapper(),
        pip_requirements=[
            "mlflow",
            "google-generativeai",
            "pandas",
            "cloudpickle==2.2.1"
        ],  
        # ✅ This replaces conda_env
        input_example=input_df,
        signature=signature
    )

    run_id = run.info.run_id
    print(f"✅ Logged model in run ID: {run_id}")

    model_uri = f"runs:/{run_id}/chatbot_model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="MedicalChatbotModel"
    )
    print(f"✅ Registered model 'MedicalChatbotModel', version: {registered_model.version}")


# COMMAND ----------

# DBTITLE 1,Local Testing
# Cell / test_script.py

# from chatbot_model import MedicalChatbot, ChatbotWrapper

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
