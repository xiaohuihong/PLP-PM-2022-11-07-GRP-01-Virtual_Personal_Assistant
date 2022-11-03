from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import re
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


class summarization_model():
    def __init__(self):
        self.config = T5Config.from_pretrained(dir_path + "/model/t5_model")
        self.model = T5ForConditionalGeneration.from_pretrained(dir_path + "/model/t5_model", config=self.config)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
        self.model.eval()

    def predict(self, context):
        tokenized = self.tokenizer([context], return_tensors='pt')
        out = self.model.generate(**tokenized, max_length=128)
        summary = self.tokenizer.decode(out[0])
        return re.search('<pad>\s(.*)</s>', summary, re.IGNORECASE).group(1)


if __name__ == '__main__':
    document = 'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.\nRepair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.\nTrains on the west coast mainline face disruption due to damage at the Lamington Viaduct.\nMany businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.\nFirst Minister Nicola Sturgeon visited the area to inspect the damage.\nThe waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.\nJeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.\nHowever, she said more preventative work could have been carried out to ensure the retaining wall did not fail.\n"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\'re neglected or forgotten," she said.\n"That may not be true but it is perhaps my perspective over the last few days.\n"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"\nMeanwhile, a flood alert remains in place across the Borders because of the constant rain.\nPeebles was badly hit by problems, sparking calls to introduce more defences in the area.\nScottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.\nThe Labour Party\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.\nHe said it was important to get the flood protection plan right but backed calls to speed up the process.\n"I was quite taken aback by the amount of damage that has been done," he said.\n"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."\nHe said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.\nHave you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled'
    document = "summarize: " + document
    model = summarization_model()
    summary = model.predict(document)
    print(f'Output:\n{summary}'.format(summary=summary))





