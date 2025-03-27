import os
import requests
import pandas as pd
import time
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
from fastapi.responses import FileResponse

BRIGHTCOVE_ACCOUNT_ID = os.getenv("BRIGHTCOVE_ACCOUNT_ID")
BRIGHTCOVE_CLIENT_ID = os.getenv("BRIGHTCOVE_CLIENT_ID")
BRIGHTCOVE_CLIENT_SECRET = os.getenv("BRIGHTCOVE_CLIENT_SECRET")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionRequest(BaseModel):
    videoId: str

def get_brightcove_token():
    url = "https://oauth.brightcove.com/v4/access_token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = (BRIGHTCOVE_CLIENT_ID, BRIGHTCOVE_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, auth=auth, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def get_video_source_url(video_id, token):
    url = f"https://cms.api.brightcove.com/v1/accounts/{BRIGHTCOVE_ACCOUNT_ID}/videos/{video_id}/sources"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    sources = response.json()
    mp4_sources = [s for s in sources if s.get("container") in ["MP4", "MOV"]]
    return mp4_sources[0]["src"] if mp4_sources else None

def download_and_encode_audio(video_url):
    print("🎧 Downloading audio from:", video_url)
    temp_path = f"/tmp/audio_{uuid4().hex}.mp3"
    os.system(f"ffmpeg -i '{video_url}' -vn -acodec libmp3lame -ar 44100 -ac 2 -ab 192k -f mp3 {temp_path}")
    with open(temp_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def transcribe_with_whisper(audio_b64):
    endpoint = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    print("📦 Base64 audio length:", len(audio_b64))

    payload = {
        "version": "e2f4a83f0de6f3f5a9e7e1db1cccb2a3d45c4a2301bc4863a4856d6bce15b105",  # Base64-enabled Whisper
        "input": {
            "audio": {
                "data": audio_b64
        }
    }

    print("📤 Sending payload to Replicate...")

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print("❌ Replicate API Error:", err)
        print("📄 Response body:", response.text)
        raise

    prediction = response.json()
    status = prediction["status"]
    prediction_url = prediction["urls"]["get"]

    print("⏳ Waiting for transcription to finish...")

    while status not in ["succeeded", "failed"]:
        time.sleep(5)
        poll = requests.get(prediction_url, headers=headers)
        poll.raise_for_status()
        prediction = poll.json()
        status = prediction["status"]

    if status == "succeeded":
        print("✅ Transcription complete!")
        return [
            {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "text": s["text"]
            } for s in prediction["output"]["segments"]
        ]
    else:
        print("❌ Transcription failed:", prediction)
        return []

def extract_cues(transcript):
    data = []
    for segment in transcript:
        text = segment["text"].lower()
        rpm_low = rpm_high = res_low = res_high = None
        if "rpm" in text:
            try:
                parts = text.split("rpm")[0].split()
                nums = [int(p) for p in parts if p.isdigit()]
                if len(nums) >= 2:
                    rpm_low, rpm_high = nums[-2], nums[-1]
            except:
                pass
        if "resistance" in text:
            try:
                parts = text.split("resistance")[1].split()
                nums = [int(p) for p in parts if p.isdigit()]
                if len(nums) >= 2:
                    res_low, res_high = nums[0], nums[1]
            except:
                pass
        data.append({
            "Segment Start (Seconds)": int(segment["start"]),
            "Segment End (Seconds)": int(segment["end"]),
            "RPM low": rpm_low,
            "RPM high": rpm_high,
            "Resistance Low": res_low,
            "Resistance High": res_high,
        })
    return pd.DataFrame(data)

@app.post("/api/transcribe")
async def transcribe(req: TranscriptionRequest):
    try:
        token = get_brightcove_token()
        video_url = get_video_source_url(req.videoId, token)
        if not video_url:
            return {"error": "Video not found or no MP4/MOV available."}
        audio_b64 = download_and_encode_audio(video_url)
        transcript = transcribe_with_whisper(audio_b64)
        if not transcript:
            return {"error": "Transcription failed."}
        df = extract_cues(transcript)
        filename = f"output_{uuid4().hex}.xlsx"
        filepath = f"/tmp/{filename}"
        df.to_excel(filepath, index=False)
        return {"downloadUrl": f"/api/download/{filename}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    filepath = f"/tmp/{filename}"
    return FileResponse(filepath, filename=filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
