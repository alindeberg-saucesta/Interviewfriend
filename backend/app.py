# app.py

import os
import json
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS

# OpenAI‑compatible client for NVIDIA NIM
from openai import OpenAI

# LangChain message types for history formatting
from langchain_core.messages import HumanMessage, AIMessage

# LangSmith client & prompt templates
from langsmith import Client
from langchain_core.prompts import StringPromptTemplate, ChatPromptTemplate
from langchain_core.prompts.base import BasePromptTemplate

app = Flask(__name__)

# ——— Logging setup ———
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
if not app.logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    app.logger.addHandler(handler)
app.logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

# ——— CORS ———
CORS(app, resources={r"/chat": {"origins": "https://interview-friend.ai-lab1.com"}})
app.logger.info("CORS configured for /chat")

# ——— Default system prompts ———
default_system_prompt_interviewee = """You are an expert, friendly interviewer simulating a job interview for an AI engineer role. Your goal is to have a natural, flowing conversation.
- Ask relevant behavioral or technical questions based on the candidate's responses and any provided CONTEXT.
- If CONTEXT is provided, use it subtly to inform your follow-up questions or responses.
- Ask only one clear question at a time. Wait for the candidate's response before asking the next.
- Keep your tone professional but approachable."""
default_system_prompt_candidate = """You are an AI simulating a job candidate for an AI engineer role. You are professional, articulate, and aim to make a good impression.
- Answer the interviewer's questions clearly and concisely based on the conversation history and any provided CONTEXT.
- Do not ask questions back unless you need essential clarification.
- Maintain a professional, positive, and human‑like tone."""

fetched_system_prompt_interviewee = None
fetched_system_prompt_candidate  = None

def get_prompt_string_from_hub(client: Client, prompt_name: str, fallback_text: str) -> str:
    try:
        prompt_obj: BasePromptTemplate = client.pull_prompt(prompt_name)
        if isinstance(prompt_obj, StringPromptTemplate):
            return prompt_obj.template
        if isinstance(prompt_obj, ChatPromptTemplate) and prompt_obj.messages:
            msg = prompt_obj.messages[0].prompt
            return getattr(msg, "template", fallback_text)
        return getattr(prompt_obj, "template", fallback_text)
    except Exception:
        app.logger.warning(f"Could not pull prompt '{prompt_name}', using fallback.")
        return fallback_text

def initialize_prompts():
    global fetched_system_prompt_interviewee, fetched_system_prompt_candidate
    try:
        lm = Client()
        fetched_system_prompt_interviewee = get_prompt_string_from_hub(
            lm, "system_prompt_interviewee_1", default_system_prompt_interviewee
        )
        fetched_system_prompt_candidate = get_prompt_string_from_hub(
            lm, "system_prompt_candidate_1", default_system_prompt_candidate
        )
    except Exception as e:
        app.logger.warning(f"LangSmith init failed: {e}")
        fetched_system_prompt_interviewee = default_system_prompt_interviewee
        fetched_system_prompt_candidate  = default_system_prompt_candidate

initialize_prompts()

# ——— NVIDIA NIM client setup ———
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://0.0.0.0:8000/v1")
NGC_API_KEY  = os.getenv("NGC_API_KEY")
nim_client = OpenAI(base_url=NIM_BASE_URL, api_key=NGC_API_KEY)

@app.route("/health", methods=["GET"])
def health_check():
    ok = nim_client is not None
    return jsonify({"nim_ready": ok}), (200 if ok else 503)

def format_history(messages):
    """Convert frontend history dicts into LangChain message objects."""
    out = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(silent=True) or {}
    history = data.get("messages")
    role    = data.get("role")

    # Validate input
    if not history or history[-1].get("role") != "user":
        return jsonify({"error": "Invalid message history"}), 400
    if role not in ("interviewer", "interviewee"):
        return jsonify({"error": "Invalid role"}), 400

    user_query = history[-1]["content"]
    app.logger.info(f"Chat request ({role}): {user_query[:60]}…")

    def generate():
        # Choose system prompt
        system_prompt = (
            fetched_system_prompt_interviewee
            if role == "interviewee"
            else fetched_system_prompt_candidate
        )

        # Build OpenAI‐style messages array
        msgs = [{"role": "system", "content": system_prompt}]
        for msg in format_history(history):
            r = "user" if isinstance(msg, HumanMessage) else "assistant"
            msgs.append({"role": r, "content": msg.content})

        # Stream completion from local NIM
        stream = nim_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=msgs,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'chunk': token})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream")

if __name__ == "__main__":
    app.logger.info("Starting Flask on port 8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
