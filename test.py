import os

if not os.path.exists("test/"):
    os.makedirs("test/")

with open(f"test/test.txt", "w") as text_file:
    text_file.write("res_txt")
print("done")
