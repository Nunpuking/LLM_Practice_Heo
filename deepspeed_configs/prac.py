import json

'''
prac_dict = { 
                "first": {
                    "first_one": "I am first one"
                    },
                "second": {
                    "second_one": "I am second one",
                    "second_two": "I am second two"
                    }
            }

with open("./practice.json", "w") as json_file:
    json.dump(prac_dict, json_file)
'''

with open("./practice.json", "r") as json_file:
    read_file = json.load(json_file)

print(read_file)
print(read_file["first"])
print(read_file["second"]["second_two"])

