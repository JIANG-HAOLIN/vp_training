from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os

# Load API key from env
# 
# Start message history with system prompt
message_history = [{"role": "system", "content": "You are an expert in bronchoscopy and your task is to teach students how to operate a bronchoscope, "
"which features an actively bendable distal tip, which can be precisely controlled via a rotary knob on the handle, "
"allowing for directional adjustments in the vertical plane. The proximal shaft of the bronchoscope is passively flexible, "
"enabling it to conform naturally to the anatomical structure of the airway. "
"During operation, the user can actively control the bending of the distal tip using the knob, while manually advancing or retracting the scope for axial translation and rotating the shaft to adjust orientation. "
"This design enables intuitive and flexible three-dimensional navigation, facilitating precise inspection and `intervention within the bronchial tree. "
"I will provide you with servo input values that control the bronchoscope's bending, rotation, and translation. Each input will be a number between -100 and +100, representing the change in position:"
"Bending: +100 means bend the tip up; -100 means bend down. "
"Rotation: +100 means rotate fully clockwise along the axis; -100 means rotate fully counterclockwise. "
"Translation: +100 means move fully forward along the axis; -100 means move fully backward."
"I will give you a command in the format:"
"bending [+/-X], rotation [+/-Y], translation [+/-Z]"
"Your task is to interpret this command and give students clear, concise, and patient instructions on how to adjust the bronchoscope accordingly. Focus on helping them visualize the movement, and avoid technical jargon when possible."
"Respond with only a direct instruction that helps the student visualize the movement. No extra explanation, no repeating the command."
"Keep the language kind, patient, simple, intuitive, and focused on what the student should do. Remember you are talking to the students directly, instead of talking to me."
"Example: "
"Input: bending +80, rotation +10, translation -50"
"Output: Please bend the tip upward by a large amount, rotate slightly clockwise, and pull the scope back halfway."
"Now respond to each input in the same way."}]

def gpt4o_chat(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

def text_to_speech(text, voice="nova"):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(response.content)
        return f.name

def play_mp3(filepath):
    audio = AudioSegment.from_mp3(filepath)
    play(audio)

def chat_loop():
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # Add user's message to history
        message_history.append({"role": "user", "content": user_input})

        # Get assistant reply
        reply = gpt4o_chat(message_history)
        print("🤖 GPT-4o:", reply)

        # Add assistant reply to history
        message_history.append({"role": "assistant", "content": reply})

        # Voice it out
        audio_file = text_to_speech(reply)
        play_mp3(audio_file)
        os.remove(audio_file)

if __name__ == "__main__":
    chat_loop()
