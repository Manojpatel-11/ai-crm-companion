import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="AI Agent CRM Assistant")

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the exact JSON structure we want Gemini to return
class CRMAnalysis(BaseModel):
    customer_summary: str = Field(description="A clean, 2-sentence summary of the conversation.")
    interest_level: str = Field(description="Must be strictly one of these three: 'Hot', 'Warm', or 'Cold'")
    budget: str = Field(description="Extracted budget (e.g., '25 Lakhs'). Use 'Not Mentioned' if missing.")
    location: str = Field(description="Preferred location mentioned by client. Use 'Not Mentioned' if missing.")
    next_actions: list[str] = Field(description="A list of specific next steps or follow-ups for the agent.")

class NoteInput(BaseModel):
    raw_note: str

# 👇 MAKE SURE TO GENERATE A FREE KEY AT aistudio.google.com AND PASTE IT HERE
client = genai.Client(api_key="AIzaSyBQgiwOu4VAsq2Tt6nlN7n3Pjg9eQj3HOo")

@app.post("/analyze-note", response_model=CRMAnalysis)
async def analyze_note(data: NoteInput):
    if not data.raw_note.strip():
        raise HTTPException(status_code=400, detail="Note content cannot be empty")
    
    system_instruction = (
        "You are an expert CRM assistant for real estate agents. Your job is to parse messy, "
        "informal voice notes or text summaries typed by an agent after talking to a buyer, "
        "and extract clean structured data. Be precise and objective."
    )
    
    try:
        # Call gemini-2.5-flash with a strict Pydantic enforcement structure
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=data.raw_note,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=CRMAnalysis,
                temperature=0.2
            ),
        )
        return CRMAnalysis.model_validate_json(response.text)
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"🔴 GEMINI CRASH LOG: {str(e)}")
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")