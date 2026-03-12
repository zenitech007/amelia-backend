def generate_action_steps(message: str):
    msg = message.lower()
    steps = []

    if "headache" in msg:
        steps.append("Drink a full glass of water and rest for 20 minutes.")
        steps.append("Check your blood sugar if you have a glucose monitor.")

    if "fatigue" in msg:
        steps.append("Drink at least 500ml of water within the next hour.")
        steps.append("Avoid prolonged sun exposure for the next few hours.")

    if "chest pain" in msg:
        steps.append("Stop all activity immediately.")
        steps.append("Seek emergency medical help or call emergency services.")

    return steps

# Test the function
print("Testing generate_action_steps function:")
print("Headache and fatigue:", generate_action_steps("I have a headache and feel tired"))
print("Chest pain:", generate_action_steps("My chest hurts"))
print("No symptoms:", generate_action_steps("I feel fine"))