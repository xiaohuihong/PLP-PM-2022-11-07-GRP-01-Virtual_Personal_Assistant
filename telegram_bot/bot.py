import requests
import json
import configparser as cfg

class telegram_chatbot():
    def __init__(self, config):
        self.token = self.read_token_from_config_file(config)
        self.base = "https://api.telegram.org/bot{}/".format(self.token)

    def get_updates(self, offset=None):
        url = self.base + "getUpdates?timeout=100"
        if offset:
            url = url + "&offset={}".format(offset + 1)
        r = requests.get(url)
        return json.loads(r.content)

    def send_message(self, msg, chat_id, parse_mode='text'):
        if parse_mode == 'MarkdownV2':
            url = self.base + "sendMessage?chat_id={}&parse_mode=MarkdownV2".format(chat_id)
        elif parse_mode == 'HTML':
            url = self.base + "sendMessage?chat_id={}&parse_mode=HTML".format(chat_id)
        else:
            url = self.base + "sendMessage?chat_id={}".format(chat_id)
        url = url + "&text={}".format(msg)
        if msg is not None:
            requests.get(url)

    def read_token_from_config_file(self, config):
        parser = cfg.ConfigParser()
        parser.read(config)
        return parser.get('creds', 'token')

    def send_photo(self, chat_id, image_path, image_caption=""):
        data = {"chat_id": chat_id, "caption": image_caption}
        url = self.base + "sendPhoto"
        with open(image_path, "rb") as image_file:
            requests.post(url, data=data, files={"photo": image_file})