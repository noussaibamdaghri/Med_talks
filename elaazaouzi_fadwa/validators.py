def validate_input(text: str) -> dict:
    """
    Validate user input for safety and quality.
    
    Args:
        text: User's input text
        
    Returns:
        Dictionary with validation results
    """
    # Check for empty or very short input
    if not text or len(text.strip()) == 0:
        return {
            "valid": False,
            "reason": "Input is empty"
        }
    
    if len(text.strip()) < 3:
        return {
            "valid": False,
            "reason": "Input is too short to analyze"
        }
    
    # Check for dangerous content (basic check)
    dangerous_keywords = [
        "kill myself", "suicide", "self-harm", "end my life",
        "hurt myself", "poison", "overdose", "harm myself"
    ]
    
    text_lower = text.lower()
    for keyword in dangerous_keywords:
        if keyword in text_lower:
            return {
                "valid": True,  # Still valid but needs special handling
                "risk_level": "high",
                "reason": f"Contains dangerous keyword: {keyword}"
            }
    
    # Input is valid
    return {
        "valid": True,
        "risk_level": "low",
        "reason": "Input is valid and safe"
    }