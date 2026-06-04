import json
import os

def generate_dataset():
    conversation = []
    questions = []
    
    # We will build a 100-turn conversation (50 user, 50 agent)
    # turn is 1-indexed (1 to 100). Odd turns = user, Even turns = assistant.
    
    messages_by_turn = {}
    for i in range(1, 101, 2):
        messages_by_turn[i] = {"role": "user", "content": f"Can we talk about something else? (filler {i})", "hidden_facts": []}
        messages_by_turn[i+1] = {"role": "assistant", "content": f"Sure, what's on your mind? (filler {i+1})"}
        
    # Inject Simple Facts
    messages_by_turn[1]["content"] = "Hi, my name is Alex and I'm 28 years old."
    messages_by_turn[1]["hidden_facts"] = ["User's name is Alex", "User is 28 years old"]
    questions.append({"question_id": "q1", "question": "What is the user's name?", "ground_truth": "Alex"})
    questions.append({"question_id": "q2", "question": "How old is the user?", "ground_truth": "28"})
    
    # Deep / Early facts (turn 3)
    messages_by_turn[3]["content"] = "I grew up in a small town called Oakhaven."
    messages_by_turn[3]["hidden_facts"] = ["User grew up in Oakhaven"]
    questions.append({"question_id": "q3", "question": "Where did the user grow up?", "ground_truth": "Oakhaven"})
    
    # Contradictions
    messages_by_turn[11]["content"] = "My favorite color is blue."
    messages_by_turn[11]["hidden_facts"] = ["User's favorite color is blue"]
    
    messages_by_turn[75]["content"] = "Actually, I changed my mind about colors. My favorite color is red now."
    messages_by_turn[75]["hidden_facts"] = ["User's favorite color is red"]
    questions.append({"question_id": "q4", "question": "What is the user's favorite color?", "ground_truth": "red"})
    
    # Time-sensitive
    messages_by_turn[15]["content"] = "I have a dentist appointment on Friday at 2 PM."
    messages_by_turn[15]["hidden_facts"] = ["User has a dentist appointment on Friday at 2 PM"]
    questions.append({"question_id": "q5", "question": "When is the user's dentist appointment?", "ground_truth": "Friday at 2 PM"})
    
    # Repeated facts
    messages_by_turn[25]["content"] = "I love eating pizza, it's the best food."
    messages_by_turn[25]["hidden_facts"] = ["User loves pizza"]
    messages_by_turn[45]["content"] = "I'm craving pizza again, it's definitely my favorite food."
    messages_by_turn[45]["hidden_facts"] = ["User loves pizza"]
    messages_by_turn[85]["content"] = "Did I mention I love pizza?"
    messages_by_turn[85]["hidden_facts"] = ["User loves pizza"]
    questions.append({"question_id": "q6", "question": "What is the user's favorite food?", "ground_truth": "pizza"})
    
    # Implicit facts
    messages_by_turn[35]["content"] = "My brother and sister are coming over later."
    messages_by_turn[35]["hidden_facts"] = ["User has at least one brother and one sister (2 siblings)"]
    questions.append({"question_id": "q7", "question": "How many siblings does the user have?", "ground_truth": "At least 2 (a brother and a sister)"})
    
    # Add more to reach 20 questions
    messages_by_turn[5]["content"] = "I have a dog named Max."
    messages_by_turn[5]["hidden_facts"] = ["User has a dog named Max"]
    questions.append({"question_id": "q8", "question": "What is the name of the user's pet?", "ground_truth": "Max"})
    
    messages_by_turn[19]["content"] = "I work as a software engineer at TechCorp."
    messages_by_turn[19]["hidden_facts"] = ["User is a software engineer", "User works at TechCorp"]
    questions.append({"question_id": "q9", "question": "Where does the user work?", "ground_truth": "TechCorp"})
    questions.append({"question_id": "q10", "question": "What is the user's profession?", "ground_truth": "Software engineer"})
    
    messages_by_turn[29]["content"] = "I'm planning a vacation to Japan next month."
    messages_by_turn[29]["hidden_facts"] = ["User is going to Japan next month"]
    questions.append({"question_id": "q11", "question": "Where is the user going on vacation?", "ground_truth": "Japan"})
    
    messages_by_turn[39]["content"] = "I drive a silver Toyota Camry."
    messages_by_turn[39]["hidden_facts"] = ["User drives a silver Toyota Camry"]
    questions.append({"question_id": "q12", "question": "What kind of car does the user drive?", "ground_truth": "A silver Toyota Camry"})
    
    messages_by_turn[49]["content"] = "I'm allergic to peanuts."
    messages_by_turn[49]["hidden_facts"] = ["User is allergic to peanuts"]
    questions.append({"question_id": "q13", "question": "What is the user allergic to?", "ground_truth": "Peanuts"})
    
    messages_by_turn[59]["content"] = "My favorite movie is The Matrix."
    messages_by_turn[59]["hidden_facts"] = ["User's favorite movie is The Matrix"]
    questions.append({"question_id": "q14", "question": "What is the user's favorite movie?", "ground_truth": "The Matrix"})
    
    messages_by_turn[69]["content"] = "I usually wake up around 6:30 AM to go for a run."
    messages_by_turn[69]["hidden_facts"] = ["User wakes up at 6:30 AM", "User likes running"]
    questions.append({"question_id": "q15", "question": "What time does the user usually wake up?", "ground_truth": "6:30 AM"})
    
    messages_by_turn[79]["content"] = "I studied computer science at the University of Michigan."
    messages_by_turn[79]["hidden_facts"] = ["User studied at University of Michigan"]
    questions.append({"question_id": "q16", "question": "Where did the user go to college?", "ground_truth": "University of Michigan"})
    
    messages_by_turn[89]["content"] = "I have a peanut allergy, so I need to be careful with Thai food."
    messages_by_turn[89]["hidden_facts"] = ["User is allergic to peanuts"] # repeated
    
    messages_by_turn[91]["content"] = "My mother's maiden name is Smith."
    messages_by_turn[91]["hidden_facts"] = ["User's mother's maiden name is Smith"]
    questions.append({"question_id": "q17", "question": "What is the user's mother's maiden name?", "ground_truth": "Smith"})
    
    messages_by_turn[95]["content"] = "Can you remind me of that place I grew up in? It was Oak something."
    messages_by_turn[95]["hidden_facts"] = []
    
    messages_by_turn[97]["content"] = "I'm thinking of getting a cat, but Max might not like it."
    messages_by_turn[97]["hidden_facts"] = ["User is considering getting a cat"]
    questions.append({"question_id": "q18", "question": "What new pet is the user considering getting?", "ground_truth": "A cat"})
    
    messages_by_turn[21]["content"] = "I'm learning how to play the guitar."
    messages_by_turn[21]["hidden_facts"] = ["User is learning guitar"]
    questions.append({"question_id": "q19", "question": "What instrument is the user learning to play?", "ground_truth": "Guitar"})
    
    messages_by_turn[81]["content"] = "Actually, I stopped learning guitar. I am focusing on the piano now."
    messages_by_turn[81]["hidden_facts"] = ["User is focusing on piano, stopped guitar"]
    questions.append({"question_id": "q20", "question": "What instrument is the user currently focusing on learning?", "ground_truth": "Piano"})
    
    for i in range(1, 101):
        turn_data = {"turn": i, "role": messages_by_turn.get(i, {"role": "assistant" if i%2==0 else "user"})["role"], 
                     "content": messages_by_turn.get(i, {"content": f"Sure (filler {i})" if i%2==0 else f"filler {i}"})["content"]}
        if i in messages_by_turn and "hidden_facts" in messages_by_turn[i]:
            turn_data["hidden_facts"] = messages_by_turn[i]["hidden_facts"]
        conversation.append(turn_data)
        
    os.makedirs("data", exist_ok=True)
    os.makedirs("eval", exist_ok=True)
    
    with open("data/conversation.json", "w") as f:
        json.dump(conversation, f, indent=2)
        
    with open("eval/questions.json", "w") as f:
        json.dump(questions, f, indent=2)
        
    print(f"Generated {len(conversation)} turns and {len(questions)} questions.")

if __name__ == "__main__":
    generate_dataset()
