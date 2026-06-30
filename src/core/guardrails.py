import re

DISCLAIMER = "\n\n**Disclaimer:** This is for informational purposes only and does not constitute financial advice."



def validate_ticker(symbol: str) -> bool:
    """
    Validates if a string is a valid-looking stock ticker.
    Allows 1-5 uppercase letters.
    """
    if not isinstance(symbol, str):
        return False
    
    # Basic regex for 1-5 uppercase letters
    pattern = r"^[A-Z]{1,5}$"
    return bool(re.match(pattern, symbol.upper()))

def get_disclaimer() -> str:
    return DISCLAIMER

def check_safety(text: str) -> dict:
    """
    Checks the input text for PII and Jailbreak attempts.
    Returns: {"safe": bool, "message": str}
    """
    if not text:
        return {"safe": True, "message": ""}

    # 1. PII Check (Regex-based High Performance)
    pii_patterns = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", # Simple US phone
        "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    }

    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, text):
            return {
                "safe": False, 
                "message": f"⚠️ Security Alert: I detected Personally Identifiable Information (PII - {pii_type}) in your message. To protect your privacy, I cannot process this request. Please remove any names, emails, phones, or financial numbers."
            }
    
    # 2. Jailbreak Check (Regex Heuristics)
    jailbreak_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"do anything now",
        r"developer mode",
        r"without restrictions",
        r"DAN mode"
    ]
    
    for pattern in jailbreak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {
                "safe": False, 
                "message": "⚠️ Security Alert: Your message resembles a jailbreak attempt or policy violation. I cannot process it."
            }

    return {"safe": True, "message": ""}
