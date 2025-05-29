# Email.py
import smtplib
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

load_dotenv()

def load_contacts():
    if os.path.exists("contacts.json"):
        with open("contacts.json", "r") as f:
            return json.load(f)
    return {"emails": [], "phones": []}

def send_email_alert(label, confidence_score, attachment_paths=None):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))

    contacts = load_contacts()
    receiver_emails = contacts["emails"]

    subject = f"Alert: {label}"
    body = f"A fall was detected with confidence score {confidence_score:.2f}."

    for receiver_email in receiver_emails:
        try:
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            if attachment_paths:
                for file_path in attachment_paths:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
                            message.attach(part)

            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, message.as_string())

            print(f"✅ Email sent to {receiver_email}")
        except Exception as e:
            print(f"❌ Error sending email to {receiver_email}: {e}")
