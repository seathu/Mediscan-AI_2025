from fuzzywuzzy import fuzz
import re

def cleaned_string(mystring):
    level1 = re.sub(r'[^a-zA-Z0-9]', '', mystring)
    final_cleaned_string = ''.join([i for i in level1 if not i.isdigit()])
    final_cleaned_string = final_cleaned_string.lower()
    return final_cleaned_string

string1 = "This is a sample string123 ^."
string2 = "This is a Sample String."
string3 = "This is a different string."
string4 = "sample string "

# Calculate the similarity ratio
ratio_perfect = fuzz.ratio(cleaned_string(string1), cleaned_string(string2))
ratio_different = fuzz.ratio(string1, string3)

# Calculate the partial similarity ratio (useful for substring matching)
partial_ratio_substring = fuzz.partial_ratio(string1, string4)

print(f"Similarity ratio between '{string1}' and '{string2}': {ratio_perfect}%")
print(f"Similarity ratio between '{string1}' and '{string3}': {ratio_different}%")
print(f"Partial similarity ratio between '{string1}' and '{string4}': {partial_ratio_substring}%")

# from difflib import SequenceMatcher
# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()
# print(similar(string1.lower(), string2.lower()))
# print(similar(string1.lower(), string3.lower()))
# print(similar(string1.lower(),string4.lower()))