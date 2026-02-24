import time
import requests
import json
from tqdm import tqdm
import threading

class APIModel:

    def __init__(self, model, api_key, api_url) -> None:
        self.__api_key = api_key
        self.__api_url = api_url
        self.model = model
        
    def __req(self, text, temperature, max_try = 5):
        url = f"{self.__api_url}"
        pay_load_dict = {
            "model": f"{self.model}",
            "messages": [{
                "role": "user",
                "content": f"{text}"
            }],
            "temperature": temperature
        }
        payload = json.dumps(pay_load_dict)
        headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {self.__api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
        }
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code != 200:
                try:
                    print(f"API error {response.status_code}: {response.text}")
                except Exception:
                    pass
                return None
            data = response.json()
            if isinstance(data, dict) and 'choices' in data and data['choices']:
                return data['choices'][0]['message']['content']
            return None
        except Exception:
            for _ in range(max_try):
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    if response.status_code != 200:
                        try:
                            print(f"API error {response.status_code}: {response.text}")
                        except Exception:
                            pass
                        time.sleep(0.2)
                        continue
                    data = response.json()
                    if isinstance(data, dict) and 'choices' in data and data['choices']:
                        return data['choices'][0]['message']['content']
                except Exception:
                    pass
                time.sleep(0.2)
            return None
    
    def chat(self, text, temperature=1):
        response = self.__req(text, temperature=temperature, max_try=5)
        return response

    def __chat(self, text, temperature, res_l, idx):
        
        response = self.__req(text, temperature=temperature)
        res_l[idx] = response
        return response
        
    def batch_chat(self, text_batch, temperature=0):
        max_threads=15 # limit max concurrent threads using model API
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self.__chat, args=(text, temperature, res_l, i))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads: 
                for t in thread_l:
                    if not t .is_alive():
                        thread_l.remove(t)
                time.sleep(0.3) # Short delay to avoid busy-waiting

        for thread in tqdm(thread_l):
            thread.join()
        return res_l
