import re

responses = {
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! How can I help you?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?" ,
    "what is your name": "I'm a chatbot created to assist you.",
    "help": "Sure, I'm here to help. What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "default": "I'm not sure I understand. Could you please rephrase?"
}

def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    
    for keyword in responses:
        if keyword == "default":
            continue
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input):
            return responses[keyword]
    
    return responses["default"]

def chatbot():
    print("Chatbot: Hello! I'm here to assist you. (type 'bye' to exit)")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower().strip() == 'bye':
            print(f"Chatbot: {responses['bye']}")
            break
        
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
