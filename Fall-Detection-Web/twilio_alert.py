# twilio_alert.py
from dotenv import load_dotenv
import os
from twilio.rest import Client
import json

load_dotenv()

def load_contacts():
    if os.path.exists("contacts.json"):
        with open("contacts.json", "r") as f:
            return json.load(f)
    return {"emails": [], "phones": []}

def send_sms_alert(message_body="Fall Detected - Check Email for more information"):
    sender = os.getenv("SENDER_NUMBER")
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    client = Client(account_sid, auth_token)

    contacts = load_contacts()
    receivers = contacts["phones"]

    if not sender or not receivers:
        print("❌ Missing SMS sender or receivers.")
        return False

    for to_number in receivers:
        try:
            msg = client.messages.create(
                from_=sender,
                body=message_body,
                to=to_number
            )
            print(f"✅ SMS sent to {to_number} (SID: {msg.sid})")
        except Exception as e:
            print(f"❌ Failed to send SMS to {to_number}: {e}")
    return True
