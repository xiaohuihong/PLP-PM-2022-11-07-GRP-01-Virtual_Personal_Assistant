from bot import telegram_chatbot
from chatterbot import ChatBot
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_first_response
from chatterbot.trainers import ChatterBotCorpusTrainer
from news_scrape.scrapeCNN import news_crawler
import re

bot = telegram_chatbot("config.cfg")

chatbot = ChatBot(
    'plp_newsbot',
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": LevenshteinDistance,
            "response_selection_method": get_first_response
        }
    ]
)

trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

def make_reply(msg):
    replies = []
    parse_mode = 'text'
    num_news = 5
    if msg is not None:
        if "push news" in msg.lower():
            print("Starting Crawling...")
            num = re.findall(r'\d+', msg)
            if num is not None and len(num) > 0:
                num_news = num[-1]
            crawler = news_crawler(num_news)
            df = crawler.get_news_df()
            for index, row in df.iterrows():
                if index == 0:
                    replies.append("Today's News:")
                replies.append("[{title}]({url})".format(title=row['title'],url=row['url']))
            parse_mode = 'MarkdownV2'
        else:
            replies.append(chatbot.get_response(msg))
    print(replies)
    return replies, parse_mode

print("Now your chatbot is alive...")

update_id = None
while True:
    updates = bot.get_updates(offset=update_id)
    updates = updates["result"]
    if updates:
        for item in updates:
            update_id = item["update_id"]
            try:
                message = str(item["message"]["text"])
            except:
                message = None
            from_ = item["message"]["from"]["id"]
            replies, parse_mode = make_reply(message)
            for reply in replies:
                bot.send_message(reply, from_, parse_mode)