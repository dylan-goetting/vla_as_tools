from vlm import GeminiVLM
import json

agent = GeminiVLM(model="gemini-1.5-pro")
tasks = json.load(open("tasks.json"))
for task, lst in tasks.items():
    print(task, len(lst))
    