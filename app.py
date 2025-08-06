from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import openai
import requests

app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def transcribe_audio_whisper(audio_path):
    result = subprocess.run([
        "whisper", audio_path, 
        "--language", "auto", 
        "--model", "base", 
        "--output_format", "txt", 
        "--output_dir", os.path.dirname(audio_path)
    ], capture_output=True)

    if result.returncode != 0:
        return None, result.stderr.decode()

    txt_path = audio_path.replace(".wav", ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        return transcript, None
    return None, "Transcript file not found"

def summarize_transcript(transcript):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a multilingual meeting assistant. Given a transcript, extract and group all important points, tasks with deadlines and people responsible, technical discussions, inquiries, meetings, and documentation needs. Return a well-structured response in the same language as the transcript."
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error from AI: {response.text}"

@app.route("/", methods=["POST"])
def handle_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        file.save(audio_path)

        transcript, error = transcribe_audio_whisper(audio_path)
        if error:
            return jsonify({"error": error}), 500

        summary = summarize_transcript(transcript)
        return jsonify({"transcript": transcript, "summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
