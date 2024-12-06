import requests
from vlm import GeminiVLM

def send_instruction(instruction):
    try:
        response = requests.post(
            "http://localhost:5000/update_instruction",
            json={"instruction": instruction}
        )
        if response.status_code == 200:
            print(f"Instruction sent: {instruction}")
        else:
            print(f"Failed to send instruction: {response.status_code}")
    except Exception as e:
        print(f"Error sending instruction: {e}")

def get_instruction():
    try:
        response = requests.get("http://localhost:5000/get_instruction")
        if response.status_code == 200:
            return response.json()["instruction"]
        else:
            print(f"Failed to get instruction: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting instruction: {e}")
        return None

class Feedback():

    def __init__(self, model):
        self.model = model
    
    def get_feedback(self, instruction):
        pass

class Agent():

    def __init__(self, model, feedback: Feedback):
        self.model = model
        self.feedback = feedback

    def split_instruction(self, instruction):
        self.model.call(images=[], text_prompt=instruction)

    def run(self):
        while True:
            instruction = get_instruction()
            if instruction is not None and instruction is not 'default':
                print(f"Received instruction: {instruction}")
                break

        self.split_instruction(instruction)

    agent = GeminiVLM(model="gemini-1.5-pro")    
    while True:
        instruction = input("Enter instruction\n").strip()
        send_instruction(instruction)

if __name__ == "__main__":
    model = GeminiVLM(model="gemini-1.5-pro")
    agent = Agent(model)
    agent.run()
