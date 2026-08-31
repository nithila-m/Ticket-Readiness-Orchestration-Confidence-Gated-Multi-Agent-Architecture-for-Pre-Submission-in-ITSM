"""
Isolated Groq connectivity test.

Loads credentials from .env and makes ONE chat completion call.
Not wired into FastAPI or LangGraph — this is purely a sanity check
so that credential and endpoint issues are debugged in isolation,
not tangled with orchestration bugs later.

Run: python test_groq.py
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq


REQUIRED_VARS = [
    "GROQ_API_KEY",
    "GROQ_MODEL",
]


def load_and_validate_env() -> dict:
    """Load .env and confirm all required variables are present.

    Prints which variables are missing (never their values), and exits
    with a non-zero status if any are missing.
    """
    load_dotenv()

    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for v in missing:
            print(f"  - {v}")
        print("\nCheck your .env file at tro-backend/.env")
        sys.exit(1)

    # Return values, but do NOT print the API key anywhere.
    return {v: os.getenv(v) for v in REQUIRED_VARS}


def make_test_call(config: dict) -> str:
    """Make one small chat completion call and return the response text."""
    client = Groq(api_key=config["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=config["GROQ_MODEL"],
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: TRO connectivity OK."},
        ],
        max_tokens=200,  #512 for initial integration
        temperature=0,
    )

    print("DEBUG RESPONSE:")
    print(response)

    print("\nDEBUG MESSAGE:")
    print(response.choices[0].message)

    print("\nDEBUG CONTENT:")
    print(repr(response.choices[0].message.content))

    return response.choices[0].message.content or ""



def main() -> None:
    config = load_and_validate_env()

    # Safe-to-print confirmation — model name only, no key.
    print(f"Model: {config['GROQ_MODEL']}")
    print("Calling Groq...\n")

    try:
        reply = make_test_call(config)
        print("Model response:")
        print(reply)
    except Exception as e:
        # Print exception type and message but not the config
        print("ERROR: Groq call failed.")
        print(f"  Type:    {type(e).__name__}")
        print(f"  Message: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()