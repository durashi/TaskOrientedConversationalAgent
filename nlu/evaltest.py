import csv
import pandas as pd

from rasa.test import get_evaluation_metrics
from pipeline_ensemble import test_stack_pipelines,test_pipelines

# itargets = ('greeting', 'greeting', 'greeting', 'confirm_answer', 'deny', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'greeting', 'confirm_answer', 'deny', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'greeting', 'confirm_answer', 'deny', 'thanks', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'confirm_answer', 'confirm_answer', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'confirm_answer', 'deny', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime')
# ipredictions =  ('greeting', 'nlu_fallback', 'greeting', 'deny', 'deny', 'deny', 'greeting', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'nlu_fallback', 'inform', 'nlu_fallback', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'nlu_fallback', 'starttime', 'nlu_fallback', 'nlu_fallback', 'theater', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'nlu_fallback', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'inform', 'deny', 'nlu_fallback', 'deny', 'nlu_fallback', 'inform', 'theater', 'inform', 'inform', 'nlu_fallback', 'inform', 'inform', 'theater', 'theater', 'ticket', 'theater', 'theater', 'ticket', 'nlu_fallback', 'nlu_fallback', 'nlu_fallback', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'nlu_fallback', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'greeting', 'greeting', 'nlu_fallback', 'nlu_fallback', 'greeting', 'nlu_fallback', 'ticket', 'inform', 'nlu_fallback', 'inform', 'nlu_fallback', 'inform', 'nlu_fallback', 'nlu_fallback', 'theater', 'theater', 'theater', 'ticket', 'inform', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'theater+starttime', 'starttime', 'greeting', 'greeting', 'deny', 'nlu_fallback', 'deny', 'thanks', 'inform', 'greeting', 'nlu_fallback', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'nlu_fallback', 'nlu_fallback', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'nlu_fallback', 'nlu_fallback', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'nlu_fallback', 'inform', 'deny', 'nlu_fallback', 'inform', 'inform', 'inform', 'inform', 'nlu_fallback', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'nlu_fallback', 'theater', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'nlu_fallback', 'starttime', 'theater+starttime', 'theater+starttime', 'theater', 'nlu_fallback', 'date+starttime', 'date+starttime')

df = pd.read_csv('data\\test_data.csv')
print(df)
sentences = df.Sentence.to_list()
itargets = df.intent.to_list()
# ipredictions = test_pipelines(sentences, "models\\nlu-20201118-145024.tar.gz")
ipredictions = test_stack_pipelines(sentences)
print(len(ipredictions), len(itargets))
get_evaluation_metrics(itargets,ipredictions)