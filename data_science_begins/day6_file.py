# with open("sample.txt","w") as file:
#     #  content = file.read()
#     file.write("hello, world")
#     file.writelines(["Alice", "Bob","Cherry"])
    

try:
    with open("sample.txt","r") as file:
         content = file.read()
except  FileNotFoundError:
    print("File Not Found!")           