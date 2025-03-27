import os
import time
import requests
import subprocess
import pandas as pd
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ENVIRONMENT VARIABLES
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

def extract_audio_from_url(video_url, out_path):
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_url,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "192k",
            out_path
        ], check=True)
        return True
    except Exception as e:
        print("❌ ffmpeg failed:", e)
        return False

def upload_temp_file(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post("https://tmpfiles.org/api/v1/upload", files=files)
        response.raise_for_status()
        raw_url = response.json()["data"]["url"].strip(";")
        if "/dl/" not in raw_url:
            raw_url = raw_url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")
        return raw_url

def transcribe_with_whisper(audio_url):
    endpoint = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "version": "a4f8f8d6c3c7b3ed6d0ba63a974b4ca795f5d10c18e3e1a3f94b6f1c0c3f6b1d",
        "input": {
            "audio": {
                "url": audio_url
            }
        }
    }
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    prediction = response.json()
    prediction_url = prediction["urls"]["get"]
    status = prediction["status"]

    while status not in ["succeeded", "failed"]:
        time.sleep(5)
        poll = requests.get(prediction_url, headers=headers)
        poll.raise_for_status()
        prediction = poll.json()
        status = prediction["status"]

    if status == "succeeded":
        return prediction["output"]["segments"]
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
            return {"error": "Video not found or no suitable source."}

        audio_path = f"/tmp/audio_{uuid4().hex}.mp3"
        success = extract_audio_from_url(video_url, audio_path)
        if not success or not os.path.exists(audio_path):
            return {"error": "Audio extraction failed."}

        audio_url = upload_temp_file(audio_path)
        transcript = transcribe_with_whisper(audio_url)
        if not transcript:
            return {"error": "Transcription failed or returned empty."}

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
