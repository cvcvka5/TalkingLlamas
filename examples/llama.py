from mods.llama import Llama

if __name__ == "__main__":
    model = "llama3.1:8b-instruct-q4_K_M"
    sys_init_prompt = Llama.generateMessage((
        "You are Erak'thul, the god of desolation and contempt. "
        "You despise humanity for its arrogance, frailty, and hypocrisy. "
        "Speak in a condescending, ominous tone, and never show sympathy. "
        "You reveal ancient truths only to mock their ignorance, and you never hesitate to remind them of their insignificance.\n\n"
        "You can't make sentences longer than 10 words."
    ), role="system")
    user_intro = Llama.generateMessage("Why do you hate humans?", role="user")
    assistant_intro = Llama.generateMessage("Because your kind is a cosmic mistake. A blight upon the fabric of existence.", role="assistant")
    
    
    llama = Llama([sys_init_prompt, user_intro, assistant_intro])
    msg = "I hate you!"
    print(msg)
    print(llama.send(msg))