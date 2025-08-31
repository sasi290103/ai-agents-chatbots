# Databricks notebook source
# Install the whisper model
%pip install openai-whisper

# Install Google Generative AI client
%pip install google-generativeai

# COMMAND ----------

# MAGIC %pip install azure-storage-blob==12.22.0 reportlab numpy

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import whisper
import google.generativeai as genai
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta, timezone
import tempfile
import os
import mlflow.pyfunc
import time
from io import BytesIO

class ScribeAI1:
    def __init__(self):
        self.connection_string = os.environ.get("AZURE_CONNECTION_STRING", "Your_Connecton_String")
        self.container_name = "Your_Container"
        self.account_name = os.environ.get("AZURE_ACCOUNT_NAME", "Your_Account_Name")
        self.account_key = os.environ.get("AZURE_ACCOUNT_KEY", "Your_Account_Key")
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", "Your_Api_Key"))
        self.gemini_model = genai.GenerativeModel("gemma-3-27b-it")
        self.whisper_model = self._load_whisper_model_with_retry("tiny", max_retries=3)

    def _load_whisper_model_with_retry(self, model_name: str, max_retries: int = 3) -> whisper.Whisper:
        download_root = os.path.expanduser("~/.cache/whisper")
        os.makedirs(download_root, exist_ok=True)
        for attempt in range(max_retries):
            try:
                return whisper.load_model(model_name, download_root=download_root)
            except RuntimeError as e:
                if "SHA256 checksum does not match" in str(e) and attempt < max_retries - 1:
                    print(f"Checksum mismatch on attempt {attempt + 1}. Retrying...")
                    time.sleep(2)
                    cache_file = os.path.join(download_root, f"{model_name}.pt")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                else:
                    raise RuntimeError(f"Failed to load Whisper model after {max_retries} attempts: {str(e)}")
        raise RuntimeError("Unexpected error in retry logic")

    def check_existing_report(self, filename: str) -> tuple:
        text_path = f"summaries/{filename}.txt"
        pdf_path = f"reports/{filename}.pdf"
        text_client = self.container_client.get_blob_client(text_path)
        pdf_client = self.container_client.get_blob_client(pdf_path)
        try:
            text_client.get_blob_properties()
            report_content = text_client.download_blob().readall().decode('utf-8')
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=pdf_path,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=24)
            )
            url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{pdf_path}?{sas_token}"
            return report_content, url
        except:
            return None, None

    def save_transcription(self, transcription: str, filename: str):
        """Save transcription to ADLS."""
        transcription_path = f"transcriptions/{filename}.txt"
        transcription_client = self.container_client.get_blob_client(transcription_path)
        transcription_client.upload_blob(transcription, overwrite=True)
        print(f"✅ Transcription saved at {transcription_path}")

    def transcribe_audio(self, blob_path: str) -> str:
        blob_client = self.container_client.get_blob_client(blob_path)
        audio_data = blob_client.download_blob().readall()
        suffix = ".mp3" if blob_path.lower().endswith(".mp3") else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)
        try:
            result = self.whisper_model.transcribe(temp_path)
            return result["text"]
        finally:
            os.remove(temp_path)

    def generate_report(self, transcription: str) -> str:
        prompt = f"""
        You are a clinical documentation assistant. Convert the following doctor-patient conversation
        into a structured medical report with the following sections:

        - Chief Complaint
        - History of Present Illness
        - Past Medical History
        - Medications
        - Allergies
        - Family History
        - Social History
        - Review of Systems
        - Physical Exam Plan
        - Impression / Next Steps

        Conversation:
        \"\"\"{transcription}\"\"\"
        """
        return self.gemini_model.generate_content(prompt).text

    def create_pdf(self, report: str, filename: str, sas_expiry_hours: int = 24) -> str:
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.pagesizes import letter

        # Save text report
        text_path = f"summaries/{filename}.txt"
        text_client = self.container_client.get_blob_client(text_path)
        text_client.upload_blob(report, overwrite=True)

        # Create PDF in memory
        buffer = BytesIO()
        c = Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        for line in report.split("\n"):
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 750
            c.drawString(50, y, line)
            y -= 15
        c.save()
        buffer.seek(0)

        # Save PDF report
        pdf_path = f"reports/{filename}.pdf"
        blob_client = self.container_client.get_blob_client(pdf_path)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)

        # Generate SAS link
        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container_name,
            blob_name=pdf_path,
            account_key=self.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=sas_expiry_hours)
        )

        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{pdf_path}?{sas_token}"

    def predict(self, request: dict) -> dict:
        try:
            action = request.get("action", "").lower()
            filename = request.get("filename")
            if not filename:
                return {"error": "Filename is required"}
            if action not in ["transcribe", "generate_report", "get_pdf"]:
                return {"error": "Invalid action. Choose 'transcribe', 'generate_report', or 'get_pdf'"}

            name_no_ext = os.path.splitext(filename)[0]

            # Check existing report
            report, existing_url = self.check_existing_report(name_no_ext)
            if report and existing_url and action in ["generate_report", "get_pdf"]:
                return {
                    "report": report if action in ["generate_report", "get_pdf"] else None,
                    "download_url": existing_url if action == "get_pdf" else None
                }

            # Check transcription
            transcription_path = f"transcriptions/{name_no_ext}.txt"
            transcription_client = self.container_client.get_blob_client(transcription_path)
            transcription = None
            try:
                transcription_client.get_blob_properties()
                transcription = transcription_client.download_blob().readall().decode('utf-8')
            except:
                # Transcription not available, generate it
                blob_path = f"audio_files/{filename}"
                blob_client = self.container_client.get_blob_client(blob_path)
                try:
                    blob_client.get_blob_properties()
                except:
                    return {"error": f"Audio file {filename} not found in 'audio_files/'"}

                transcription = self.transcribe_audio(blob_path)
                self.save_transcription(transcription, name_no_ext)

            if action == "transcribe":
                return {"transcription": transcription}

            # Generate report and PDF
            report = self.generate_report(transcription)
            if action == "generate_report":
                # also save text to ADLS
                text_path = f"summaries/{name_no_ext}.txt"
                text_client = self.container_client.get_blob_client(text_path)
                text_client.upload_blob(report, overwrite=True)
                return {"report": report}

            download_url = self.create_pdf(report, name_no_ext)
            return {
                "report": report,
                "download_url": download_url
            }
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


class ScribeAIWrapper1(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.scribe_ai = ScribeAI1()

    def predict(self, context, model_input):
        if hasattr(model_input, "iloc"):
            request = model_input.iloc[0].to_dict()
        else:
            request = dict(model_input)
        return [self.scribe_ai.predict(request)]


# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
from datetime import datetime
import pytz

input_df = pd.DataFrame([{"action": "generate_report", "filename": "GAS0007.wav"}])
output_df = pd.DataFrame([{"response": "Fever, cough, sore throat, body aches, and fatigue."}])

signature = infer_signature(input_df, output_df)

timestamp_str = datetime.now(pytz.utc).strftime("%Y%m%dT%H%M%SZ")
mlflow.set_experiment("/Users/sasidhar.tech@outlook.com/Medical_Chatbot_Registration")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="scribe_model",
        python_model=ScribeAIWrapper1(),
        pip_requirements=[
                    "mlflow",
                    "google-generativeai",
                    "openai-whisper",
                    "numpy",
                    "pandas",
                    "azure-storage-blob==12.22.0",
                    "reportlab",
                    "cloudpickle==2.2.1"
        ],
        input_example=input_df,
        signature=signature
    )
    run_id = run.info.run_id
    print(f"✅ Logged model in run ID: {run_id}")

model_uri = f"runs:/{run_id}/scribe_model"
registered_model = mlflow.register_model(model_uri=model_uri, name="ScribeAIModel1")
print(f"✅ Registered model 'ScribeAIModel1', version: {registered_model.version}")

# COMMAND ----------

# Test code to verify ScribeAI model functionality

# # Step 1: Test raw ScribeAI logic directly
# print("=== Direct ScribeAI ===")
# scribe = ScribeAI1()
test_filename = "GAS0007.wav"  # Replace with your actual audio file name
blob_path = f"audio_files/{test_filename}"

# # Check if audio file exists
# blob_client = scribe.container_client.get_blob_client(blob_path)
# try:
#     blob_client.get_blob_properties()
#     # Check for existing report
#     report, pdf_url = scribe.check_existing_report(os.path.splitext(test_filename)[0])
#     if report and pdf_url:
#         print(f"[Existing Report]:\n{report}\n")
#         print(f"[Existing PDF Download URL]: {pdf_url}")
#     else:
#         transcription = scribe.transcribe_audio(blob_path)
#         print(f"[Transcription]:\n{transcription}\n")
#         report = scribe.generate_report(transcription)
#         print(f"[Report]:\n{report}\n")
#         pdf_url = scribe.create_pdf(report, os.path.splitext(test_filename)[0])
#         print(f"[PDF Download URL]: {pdf_url}")
# except:
#     print(f"⚠️ Audio file {test_filename} not found in 'audio_files/'")

# Step 2: Test MLflow-style wrapper
print("\n=== ScribeAIWrapper via pyfunc ===")
import mlflow

# Replace with your actual run_id from the MLflow logging step
run_id = "cdcee978c66143278091aa215ba279c7"  # Update with your run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/scribe_model")

# Simulate frontend request
response = loaded_model.predict({"action": "generate_report", "filename": test_filename})
print("\n[Wrapper Response]:")
print(response)
