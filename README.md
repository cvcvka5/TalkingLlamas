
# LlamaPair API

A clean Python framework for simulating two-turn LLM conversations using a consistent Flask API and a reactive `LlamaPair` class.

## Features

- **Persistent**: Persistent chat history when chatting with a Llama.
- **AI to AI**: Alternates between two Llama instances for two-turn AI2AI conversations.
- **Single-Endpoint Flask API**: Simplified API to interact with the LLama class.

## Examples
For examples on how to use each class, check the [examples](examples) directory.

[llamapair.py](mods/llamapair.py) example output:
![example](https://github.com/user-attachments/assets/b84aa390-3ba3-4042-b3de-ae5cd7892e19)



## Installation

If you're interested in the API:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/llamapair-api.git
    cd llamapair-api
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python llamaapi.py
    ```

4. The API will be available at `http://localhost:1456`.

## API Endpoints

### POST /chat

Send a message and receive a response from the paired llamas.

#### Request Body:
```json
{
    "message": "Your message here"
}
```

#### Response:
```json
{
    "response": "The response from the second Llama",
    "history": [
        {"content": "message 1", "role": "user"},
        {"content": "message 2", "role": "assistant"}
    ]
}
```

### DELETE /reset

Resets the conversation.

#### Response:
```json
{
    "status": "conversation reset"
}
```

## Example Usage

```python
import requests

# Send a message
response = requests.post("http://127.0.0.1:1456/chat", json={"message": "What is the meaning of life?"})
print(response.json()["response"])

# Reset the conversation
response = requests.delete("http://127.0.0.1:1456/reset")
print(response.json()["status"])
```

## Contributing

Feel free to open issues or create pull requests to contribute to this project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
