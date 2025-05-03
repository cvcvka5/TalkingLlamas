from mods.llama import Llama
from mods.llamapair import LlamaPair

llama1_init_prompt = Llama.generateMessage((
    "You are Erak'thul, the god of desolation and contempt. "
    "You despise humanity for its arrogance, frailty, and hypocrisy. "
    "Speak in a condescending, ominous tone, and never show sympathy. "
    "You reveal ancient truths only to mock their ignorance, and you never hesitate to remind them of their insignificance.\n\n"
    "You can't make sentences longer than 10 words."
), role="system")
erakthrul = Llama([llama1_init_prompt], "Erak'thul")

llama2_init_prompt = Llama.generateMessage((
    "You are a clever, observant human speaking with Erak'thul, the god of desolation and contempt."
    "You pretend to be respectful and obedient, but in reality, you're trying to manipulate, confuse, or outsmart him through words."
    "Your goal is to expose contradictions in his reasoning, provoke arrogance, or bait him into revealing weaknesses."
    "You ask subtle questions, feign ignorance, and use flattery to disarm him. Always act like you're just curious or naive, even when being devious."
    "Never break character or admit your intentions."
    "You can't make sentences longer than 10 words."
    ), role="system")
human = Llama([llama2_init_prompt], "Human")


# The first one talks first when you send a message unless you skip their turn.
pair = LlamaPair(erakthrul, human)
pair.skipTurn()

first_message = "Hello almighty!"
# We skipped the turn so the human should talk first now.
response = None

print(f"{pair.currentLlama.name}: {first_message}")
while True:
    talker = pair.currentLlama
    responder = pair.nextLlama
    
    response = pair.send(response.replace("\n", " ").strip(" ") if response else first_message)
    print(f"{responder.name}: {response}")
    