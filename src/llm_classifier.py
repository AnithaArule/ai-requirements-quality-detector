import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Use the OpenAI client class directly to avoid import resolution issues
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_json_response(json_string):
    # Remove any leading/trailing whitespace
    json_string = json_string.strip()
    json_string = re.sub(r'^[^{]*', '', json_string)  # Remove any leading text before the first '{'
    json_string = re.sub(r'[^}]*$', '', json_string)  # Remove any trailing text after the last '}'
    json_string = re.sub(r'"\s*:\s*"', '": "', json_string)  # Ensure proper spacing around colons
    return json_string

def classify_requirement_with_llm(requirement_text):
    prompt = f"""
    You are an expert software requirements analyst.

Classify the following requirement into exactly ONE category:
- ambiguous
- incomplete
- unverifiable
- well-defined

Definitions:
ambiguous = vague wording or open to multiple interpretations.
incomplete = missing key information such as actor, condition, constraint, or expected outcome.
unverifiable = subjective or not objectively testable.
well-defined = clear, specific, measurable, and testable.

Do not classify a requirement as ambiguous only because it is broad.
If the requirement lacks details, classify it as incomplete.
If it contains subjective quality words such as user-friendly, intuitive, attractive, convenient, classify it as unverifiable.
If it contains measurable values, dates, timestamps, actors, or objective outputs, classify it as well-defined.

Examples:
"The system should be fast." → ambiguous
"The system generates reports." → incomplete
"The interface must be user-friendly." → unverifiable
"The system shall respond within 2 seconds." → well-defined

Return ONLY valid JSON.
Do not use markdown.
Do not wrap the response in ```json.

Use this exact schema:
{{
  "label": "{requirement_text}",
  "category": "ambiguous",
  "confidence": 0.95,
  "explanation": "short explanation",
  "suggestion": "short revision suggestion"
}}

Requirement:
"{requirement_text}"
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=250
    )

    raw_output = response.choices[0].message.content.strip()
    cleaned_output = clean_json_response(raw_output)

    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        print("Error decoding LLM output:", raw_output)
        return {
            "label": requirement_text,
            "category": "well-defined",
            "confidence": 0.0,
            "explanation": "The LLM response could not be parsed.",
            "suggestion": "Manual review is required."
        }
    