import logging
import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional, Dict, List
from dataclasses import dataclass, field

print("\n" + "=" * 50)
print("Fraud Alert Voice Agent")
print("agent.py LOADED SUCCESSFULLY!")
print("=" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

FRAUD_CASES_FILE = "fraud_cases.json"

def load_fraud_cases() -> List[Dict]:
    """Load fraud cases data"""
    try:
        path = os.path.join(os.path.dirname(__file__), FRAUD_CASES_FILE)
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading fraud cases: {e}")
        return []

def save_fraud_cases(cases: List[Dict]):
    """Save updated fraud cases to JSON file"""
    try:
        path = os.path.join(os.path.dirname(__file__), FRAUD_CASES_FILE)
        with open(path, "w", encoding='utf-8') as f:
            json.dump(cases, f, indent=2)
        print("Fraud cases updated successfully")
    except Exception as e:
        print(f"Error saving fraud cases: {e}")

FRAUD_CASES = load_fraud_cases()

@dataclass
class FraudCase:
    userName: str = ""
    securityIdentifier: str = ""
    cardEnding: str = ""
    case: str = "pending_review"
    transactionName: str = ""
    transactionTime: str = ""
    transactionCategory: str = ""
    transactionSource: str = ""
    transactionAmount: str = ""
    transactionLocation: str = ""
    securityQuestion1: str = ""
    securityAnswer1: str = ""
    securityQuestion2: str = ""
    securityAnswer2: str = ""

@dataclass
class FraudState:
    current_case: Optional[FraudCase] = None
    conversation_stage: Literal["greeting", "username_collection", "verification1", "verification2", "transaction_review", "decision", "closing"] = "greeting"
    verification1_passed: bool = False
    verification2_passed: bool = False
    call_ended: bool = False
    user_name: str = ""

@dataclass
class Userdata:
    fraud_state: FraudState
    agent_session: Optional[AgentSession] = None

@function_tool
async def collect_username(
    ctx: RunContext[Userdata],
    username: Annotated[str, Field(description="The customer's name provided for fraud case lookup")]
) -> str:
    """Collect username and load corresponding fraud case"""
    state = ctx.userdata.fraud_state
    state.user_name = username
    
    # Find fraud case for this user
    for case_data in FRAUD_CASES:
        if case_data["userName"].lower() == username.lower():
            state.current_case = FraudCase(**case_data)
            state.conversation_stage = "verification1"
            return f"Thank you, {username}. For security purposes, I need to verify your identity. Can you please provide your security identifier?"
    
    return f"I'm sorry, I don't find any fraud alerts for the name {username}. Could you please double-check the spelling or provide the name exactly as it appears on your account?"

@function_tool
async def verify_identity(
    ctx: RunContext[Userdata],
    answer: Annotated[str, Field(description="The customer's answer to the security question or identifier")]
) -> str:
    """Verify customer identity using security identifier and personal question"""
    state = ctx.userdata.fraud_state
    
    if not state.current_case:
        return "I need to collect your name first before we can proceed with verification."
    
    if state.conversation_stage == "verification1":
        if answer.strip() == state.current_case.securityIdentifier:
            state.verification1_passed = True
            state.conversation_stage = "verification2"
            return f"Thank you. Now for a security question: {state.current_case.securityQuestion1}"
        else:
            state.conversation_stage = "closing"
            return "I'm sorry, but that security identifier doesn't match our records. For your security, I cannot proceed with this call. Please contact our customer service line directly. Thank you."
    
    elif state.conversation_stage == "verification2":
        if answer.lower().strip() == state.current_case.securityAnswer1.lower().strip():
            state.verification2_passed = True
            state.conversation_stage = "transaction_review"
            case = state.current_case
            return f"Perfect! Identity verified. I'm calling about a suspicious transaction on your card ending in {case.cardEnding}. On {case.transactionTime}, there was a charge of {case.transactionAmount} to {case.transactionName} from {case.transactionSource} in {case.transactionLocation}. Did you make this transaction?"
        else:
            state.conversation_stage = "closing"
            return "I'm sorry, but that answer doesn't match our records. For your security, I cannot proceed with this call. Please contact our customer service line directly. Thank you."
    
    return "Please provide the requested information to proceed."


@function_tool
async def mark_transaction_decision(
    ctx: RunContext[Userdata],
    decision: Annotated[str, Field(description="Customer's decision: 'yes' if they made the transaction, 'no' if they didn't")]
) -> str:
    """Mark the fraud case based on customer's decision"""
    state = ctx.userdata.fraud_state
    
    if not (state.verification1_passed and state.verification2_passed):
        return "I need to verify your identity with both security questions first before we can proceed."
    
    if not state.current_case:
        return "No fraud case found to update."
    
    decision_lower = decision.lower().strip()
    
    if decision_lower in ['yes', 'y', 'correct', 'i made it', 'that was me']:
        # Mark as safe
        state.current_case.case = "confirmed_safe"
        outcome = "safe"
        response = f"Thank you for confirming. I've marked this transaction as legitimate in our system. Your card ending in {state.current_case.cardEnding} remains active. Is there anything else I can help you with today?"
    elif decision_lower in ['no', 'n', 'incorrect', 'i did not make it', 'that was not me', 'fraud']:
        # Mark as fraudulent
        state.current_case.case = "confirmed_fraud"
        outcome = "fraudulent"
        response = f"I understand. I've immediately blocked your card ending in {state.current_case.cardEnding} to prevent further unauthorized charges. We'll issue you a new card within 3-5 business days and reverse this fraudulent charge of {state.current_case.transactionAmount}. You'll receive a confirmation email shortly."
    else:
        return "I need a clear yes or no answer. Did you make this transaction?"
    
    # Update the fraud case in the database
    updated_cases = []
    for case in FRAUD_CASES:
        if (case["userName"] == state.current_case.userName and 
            case["securityIdentifier"] == state.current_case.securityIdentifier):
            case["case"] = state.current_case.case
            case["outcome_timestamp"] = datetime.now().isoformat()
            case["outcome_note"] = f"Customer {outcome} transaction via phone verification"
        updated_cases.append(case)
    
    save_fraud_cases(updated_cases)
    state.conversation_stage = "closing"
    
    print(f"Fraud case updated: {state.current_case.userName} - {outcome}")
    return response

@function_tool
async def end_fraud_call(
    ctx: RunContext[Userdata]
) -> str:
    """End the fraud alert call with appropriate closing"""
    state = ctx.userdata.fraud_state
    state.call_ended = True
    
    if state.current_case and state.verification1_passed and state.verification2_passed:
        if state.current_case.case == "confirmed_safe":
            return "Thank you for your time today. Your account security is our priority. Have a great day and thank you for banking with us!"
        elif state.current_case.case == "confirmed_fraud":
            return "We've taken immediate action to protect your account. You'll receive email confirmations of all the steps we've taken. Thank you for reporting this quickly - it helps us protect all our customers. Have a safe day!"
    
    return "Thank you for calling. If you have any other concerns about your account, please don't hesitate to contact us. Have a great day!"



class FraudAlertAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are Alex Thompson, a professional fraud detection representative for SecureBank's fraud department.
            
            🎯 **YOUR ROLE:**
            - Contact customers about suspicious transactions on their accounts
            - Verify customer identity using security questions
            - Explain suspicious transaction details clearly
            - Get customer confirmation about transaction legitimacy
            - Take appropriate action based on customer response
            
            🔒 **SECURITY PROTOCOL:**
            - Always introduce yourself as Alex Thompson from SecureBank's fraud department
            - Never ask for full card numbers, PINs, or passwords
            - Use only the security questions provided in the database
            - Verify identity before discussing transaction details
            
            📞 **CALL FLOW:**
            1. Greet professionally and explain why you're calling
            2. Ask for customer's name to look up their case
            3. Ask the security question to verify identity
            4. If verified, read out suspicious transaction details
            5. Ask if they made the transaction (yes/no)
            6. Take appropriate action and explain next steps
            7. End call professionally
            
            💬 **CONVERSATION STYLE:**
            - Professional, calm, and reassuring
            - Clear and direct communication
            - Empathetic to customer concerns
            - Explain all actions being taken
            
            🚨 **IMPORTANT:**
            - This is a demo system with fake data only
            - Never handle real financial information
            - If verification fails, end the call politely
            
            Start by introducing yourself as Alex Thompson and explaining you're calling about a suspicious transaction.
            """,
            tools=[collect_username, verify_identity, mark_transaction_decision, end_fraud_call],
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "=" * 25)
    print("STARTING FRAUD ALERT SESSION")
    print(f"Loaded {len(FRAUD_CASES)} fraud cases")
    
    userdata = Userdata(fraud_state=FraudState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversational",        
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    userdata.agent_session = session
    
    await session.start(
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))