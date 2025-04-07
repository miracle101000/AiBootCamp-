# person = {"name":"Alice","age":25,"grade":"A"}

# person["address"] = "123 Main St"

# person["age"] = 32 

# if "grade" in person:
#     del person["grade"]
    

# print(person)    


sentence = input("Enter a sentence: ")

words = sentence.split()

word_count ={}

for word in words:
    word = word.lower()
    if word in word_count:
       word_count[word] += 1
    else:
        word_count[word] = 1  
mylist = [3,2,1]

mylist.reverse()        
print(mylist)        
         