from __future__ import annotations

from typing import List, Dict, Callable
import requests
import json
import threading


class Llama:
    MODEL = "llama3.1:8b-instruct-q4_K_M"
    
    def __init__(self, messages: List[Dict[str, str]], name: str = None, onResponse: Callable = lambda msg, res: None):
        """
        Initializes the instance with a list of messages, an optional name, 
        and sets the default API endpoint for Ollama.
        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, 
                where each dictionary contains message details.
            name (str, optional): An optional name for the instance. Defaults to None.
            onResponse (Callable, optional): A callback function that is invoked with the message and response. Defaults to a no-op lambda function.
        """
        
        self.messages = messages
        self.ollama_api = "http://localhost:11434/api/chat"
        self.name = name
        self.onResponse = onResponse
    
    def send(self, message: str, role: str = "user", save: bool = True) -> str:
        """
        Sends a message to the Ollama API and processes the response.
        Args:
            message (str): The message to send to the API.
            role (str, optional): The role of the sender (e.g., "user"). Defaults to "user".
            save (bool, optional): Whether to save the message and response to the internal message history. Defaults to True.
        Returns:
            str: The full response content received from the API.
        Raises:
            requests.exceptions.RequestException: If the HTTP request to the API fails.
            json.JSONDecodeError: If the response contains invalid JSON.
        """
        
        payload = self._generate_payload(message, role)
        res = requests.post(self.ollama_api, json=payload, stream=True)

        if save:
            self.messages.append(Llama.generateMessage(message, role="user"))
        
        full_response = ""
        
        for line in res.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if 'message' in data and 'content' in data['message']:
                    chunk = data['message']['content']
                    full_response += chunk  # <-- accumulate chunks

        if save:
            self.messages.append(Llama.generateMessage(full_response, role="assistant"))

        return full_response

    def sendReactive(self, message: str, role: str = "user", save: bool = True) -> None:
        """
        Sends a message reactively in a separate thread.
        This method sends a message to the system and processes the response asynchronously
        by running the operation in a separate thread. It ensures that the main thread is
        not blocked during the execution. Then it calls the 'onResponse' function like onReponse(message, response)
        Args:
            message (str): The message to be sent.
            role (str, optional): The role of the sender. Defaults to "user".
            save (bool, optional): Whether to save the message and response. Defaults to True.
        Returns:
            None
        """
        
        def inner(message: str, role: str = "user", save: bool = True) -> None:
            response = self.send(message=message, role=role, save=save)
            self.onResponse(message, response)
            
            return None
        
        threading.Thread(target=inner, kwargs={"message": message, "role": role, "save": save}).start()

    def _generate_payload(self, message: str, role: str="user") -> Dict[str, str]:
        """
        Generates a payload dictionary for the Llama model API request.
        Args:
            message (str): The message content to be included in the payload.
            role (str, optional): The role of the message sender. Defaults to "user".
        Returns:
            Dict[str, str]: A dictionary containing the model, messages, and stream flag
            for the API request.
        """
        
        return {
            "model": Llama.MODEL,
            "messages": [
                *self.messages,
                Llama.generateMessage(message, role)
            ],
            "stream": True
        }
    
    @staticmethod
    def generateMessage(message: str, role: str = "user") -> Dict[str, str]:
        """
        Generates a message dictionary with specified content and role.
        Args:
            message (str): The content of the message.
            role (str, optional): The role associated with the message. Defaults to "user".
        Returns:
            Dict[str, str]: A dictionary containing the message content and role.
        """
        
        return {"content": message, "role": role}