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

def get_waiting():
    try:
        response = requests.get("http://localhost:5000/get_waiting")
        if response.status_code == 200:
            return response.json()["waiting"]
        else:
            print(f"Failed to get waiting: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting waiting: {e}")
        return None 

def get_latest_frame():
    try:
        response = requests.get("http://localhost:5000/get_latest_frame")
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to get latest frame: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting latest frame: {e}")
        return None

class Feedback():

    def __init__(self, model):
        self.model = model
    
    def get_feedback(self, instruction):
        # Returns yes, fail, or not_yet
        pass

    def check_stall(self, dt):
        # Returns True if the agent is stalling
        pass

class Agent():

    feedback_check_s = 5
    stall_check_s = 10

    def __init__(self, model, feedback: Feedback):
        self.model = model
        self.feedback = feedback

    def split_instruction(self, instruction, context):
        self.model.call(images=[], text_prompt=instruction)

    def run_episode(self):
        while True:
            instruction = get_instruction()
            if instruction is not None and instruction is not 'default':
                print(f"Received starting instruction: {instruction}")
                break

            time.sleep(0.2)

        start_time = time.time()
        while not get_waiting():
            context = None
            instructions = self.split_instruction(instruction, context)
            
            for instruction in instructions:
                self.send_instruction(instruction)
                dt = int(time.time() - start_time)
                if dt % self.feedback_check_s == 0:
                    feedback = self.feedback.get_feedback(instruction)
                if dt % self.stall_check_s == 0:
                    stall_check = self.feedback.check_stall(dt)
                if stall_check or feedback == "fail":
                    self.send_instruction("reset")
                    break
                    
    def run(self):
        while True:
            self.run_episode()

if __name__ == "__main__":
    model = GeminiVLM(model="gemini-1.5-pro")
    agent = Agent(model)
    agent.run()
