import json
import os
import re
import shutil
import uuid

import fitz  # PyMuPDF
import google.generativeai as genai
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PIL import Image

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SCOPES = [
    "https://www.googleapis.com/auth/forms.body"
]

# ----------- Utilities ------------

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()

def extract_text_from_image(path: str) -> str:
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def clean_gemini_json(raw: str) -> dict:
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        clean = raw[start:end]
        return json.loads(clean)
    except Exception as e:
        raise ValueError(f"Invalid Gemini JSON: {e}")


# ---------- Google Form Creation ----------

def create_google_form(quiz):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    form_service = build('forms', 'v1', credentials=creds)

    # Step 1: Create the Form
    form = form_service.forms().create(
        body={"info": {
            "title": quiz.get("title", "AI Generated Quiz"),
            "documentTitle": quiz.get("title", "AI Generated Quiz")
        }}
    ).execute()
    form_id = form["formId"]

    # Step 2: Enable quiz settings
    form_service.forms().batchUpdate(
        formId=form_id,
        body={
            "requests": [
                {
                    "updateSettings": {
                        "settings": {
                            "quizSettings": {
                                "isQuiz": True
                            }
                        },
                        "updateMask": "quizSettings.isQuiz"
                    }
                }
            ]
        }
    ).execute()

    # Step 3: Add Questions
    requests = []
    for q in quiz["questions"]:
        question_text = q["question"]
        raw_options = q["options"]
        correct = q["correct_answer"]
        difficulty = q.get("difficulty", "easy").lower()
        shuffle = q.get("shuffle", True)

        cleaned_options = [re.sub(r"^[A-Da-d][\)\.:]?\s*", "", opt.strip()) for opt in raw_options]

        # Match correct answer against cleaned options
        matched_correct = None
        for opt in raw_options:
            if correct.strip().lower() in opt.lower():
                matched_correct = re.sub(r"^[A-Da-d][\)\.:]?\s*", "", opt.strip())
                break
        if matched_correct is None:
            matched_correct = cleaned_options[0]  # fallback

        points = 1
        if difficulty == "medium":
            points = 2
        elif difficulty == "hard":
            points = 3

        request = {
            "createItem": {
                "item": {
                    "title": f"{question_text}",
                    "questionItem": {
                        "question": {
                            "required": True,
                            "grading": {
                                "pointValue": points,
                                "correctAnswers": {
                                    "answers": [{"value": matched_correct}]
                                }
                            },
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [{"value": opt} for opt in cleaned_options],
                                "shuffle": shuffle
                            }
                        }
                    }
                },
                "location": {"index": 0}
            }
        }
        requests.append(request)

    form_service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()

    final_form = form_service.forms().get(formId=form_id).execute()
    return final_form["responderUri"]


# ---------- Routes ----------

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower()
    file_id = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, file_id)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": file_id}


@app.post("/generate-quiz/")
async def generate_quiz(request: Request):
    data = await request.json()
    filename = data.get("filename")
    if not filename:
        return JSONResponse(status_code=400, content={"error": "Missing filename"})

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        content_text = extract_text_from_pdf(file_path)
    elif ext in ["jpg", "jpeg", "png"]:
        content_text = extract_text_from_image(file_path)
    else:
        return JSONResponse(status_code=415, content={"error": "Unsupported file type"})

    if not content_text.strip():
        return JSONResponse(status_code=422, content={"error": "No text found"})

    prompt = f"""
    Generate a multiple-choice quiz based on the content below. Generate between 10 to 50 questions.
    Each question must have:
    - Four options
    - The correct answer
    - A short explanation
    - A difficulty level (easy, medium, hard)

    Content:
    '''{content_text[:4000]}'''

    Output JSON format:
    {{
      "title": "Quiz Title",
      "questions": [
        {{
          "question": "Sample Question?",
          "options": ["A", "B", "C", "D"],
          "correct_answer": "B",
          "explanation": "Explanation here",
          "difficulty": "easy"
        }}
      ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        quiz_data = clean_gemini_json(response.text)
        return {"quiz": quiz_data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Gemini Error: {str(e)}"})


@app.post("/export-google-forms/")
async def export_google_form(request: Request):
    try:
        data = await request.json()
        quiz = data.get("quiz")
        if not quiz:
            return JSONResponse(status_code=400, content={"error": "Missing quiz data"})

        form_url = create_google_form(quiz)
        return {"url": form_url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Export Error: {str(e)}"})
