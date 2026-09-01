# TRO — Ticket Readiness Orchestration

A confidence-gated multi-agent architecture for pre-submission clarification, deflection, and correlation in ITSM.

## Setup

1. Create and activate a virtual environment (Python 3.11):
   \`\`\`
   py -3.11 -m venv venv
   venv\Scripts\activate
   \`\`\`
2. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
3. Copy \`.env.example\` to \`.env\` and fill in your API key.
4. Run the app:
   \`\`\`
   uvicorn app.main:app --reload
   \`\`\`
5. Run tests:
   \`\`\`
   pytest
   \`\`\`

## Status

Milestone 1: Project initialization + health check endpoint. Agent 1 (information extraction) in progress.