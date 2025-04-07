import re

# text = "Contact me at 123-456-7890"
# digits = re.findall(r"\d+",text)
# print(digits)

# updated_text = re.sub(r"\d","X",text)
# print(updated_text)


# def clean_text(text):
#     #Remove punctuation
#     text = re.sub(r"[^\w\s]","",text)
#     #Remove extra spaces
#     text =" ".join(text.split())
#     #Convert to lowercase
#     return text.lower()

# input_text ="     Hello, World.!!! Welcome to Python, Programming....   "
# cleaned_text = clean_text(input_text)

# print("Cleaned Text: ",cleaned_text)


def is_palindrome(text):
    text = "".join(char.lower() for char in text if char.isalnum())
    return text == text[::-1]


input_text = input("Enter a string: ")
if is_palindrome(input_text):
    print(f'"{input_text}" is a palindrome')
else:
    print(f'"{input_text}" is not a palindrome')    
