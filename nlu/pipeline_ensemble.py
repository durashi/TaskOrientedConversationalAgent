import logging
import typing
import glob, os
import json
from typing import Optional, Text

from rasa.shared.utils.cli import print_info, print_success
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.nlu.model import Interpreter, Metadata
from rasa import model
from rasa.shared.utils.io import json_to_string
import rasa.utils.common

if typing.TYPE_CHECKING:
    from rasa.nlu.components import ComponentBuilder

logger = logging.getLogger(__name__)


def run_cmdline(component_builder: Optional["ComponentBuilder"] = None) -> None:
    # interpreter = Interpreter.load(model_path, component_builder)
    regex_interpreter = RegexInterpreter()

    print_success("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        print_success("Next message:")
        try:
            message = input().strip()
        except (EOFError, KeyboardInterrupt):
            print_info("Wrapping up command line chat...")
            break

        if message.startswith(INTENT_MESSAGE_PREFIX):
            result = rasa.utils.common.run_in_loop(regex_interpreter.parse(message))
        else:
            # result = interpreter.parse(message)
            model_folder_path = "stack_models\pipelines"
            model_stack_predict(model_folder_path, message)

        # print(json_to_string(result))


def test_pipelines(test_data, model_path):
    model_path = get_loaded_model_path(model_path)
    interpreter = Interpreter.load(model_path, None)
    test_result = []
    for sentence in (test_data):
        result = interpreter.parse(sentence)
        test_result.append(result['intent']['name'])
    print("test result  :",test_result)
    return(tuple(test_result))


def test_stack_pipelines(test_data):
    model_folder_path = "stack_models\pipelines"
    test_result = []
    for sentence in (test_data):
        result = model_stack_predict(model_folder_path, sentence)
        test_result.append(result['intent']['name'])
    print("test result  :",test_result)
    return(tuple(test_result))
    

def get_loaded_model_path(model_path):
    temp_path = model.get_model(model_path)
    loaded_model_path = temp_path + "\\nlu"
    return loaded_model_path


def model_stack_predict(model_folder_path, message):
    models = glob.glob(model_folder_path + "\\*")
    nlu_result = {"text": message}
    intent_ranking = []
    entities = []
    for model in models:
        config = model.split("\\")[-1]
        print("config", config)
        model = max(glob.glob(model + "\\*"), key=os.path.getctime)

        with open("results\pipelines\{}\intent_report.json".format(config)) as f:
            intent_report = json.load(f)
        f1_score = intent_report["micro avg"]["f1-score"]
        print("f1_score ", f1_score)
        # f1_score = 1
        model_path = get_loaded_model_path(model)
        interpreter = Interpreter.load(model_path, component_builder=None)
        result = interpreter.parse(message)
        # print("result ", result)

        for i in range(3):
            pred_intent = result["intent_ranking"][i]
            if not (pred_intent["name"]=='nlu_fallback'):
                if not intent_ranking:
                    pred_intent["count"] = f1_score
                    pred_intent["confidence"] = pred_intent["confidence"] * f1_score
                    intent_ranking.append(pred_intent)
                else:
                    for j in intent_ranking:
                        if j["name"] == pred_intent["name"]:
                            # print("######################")
                            j["confidence"] = (
                                j["count"] * j["confidence"]
                                + pred_intent["confidence"] * f1_score
                            ) / (j["count"] + f1_score)
                            # print("#####################jconfidence ",j['confidence'])
                            j["count"] += f1_score
                            break
                    else:
                        pred_intent["count"] = 1
                        intent_ranking.append(pred_intent)
            # print("intent_ranking ", intent_ranking)

        for entity in result["entities"]:
            if not entities:
                entities.append(entity)
            elif (entity["entity"], entity["value"]) not in [
                (en["entity"], en["value"]) for en in entities
            ]:
                entities.append(entity)
            ## todo : check for same entities with diff values and same value with diff entities

    # print("final_intent_ranking ", intent_ranking)
    # print("final_intent ", [intent for intent in intent_ranking if intent['confidence']== max([en['confidence'] for en in intent_ranking])])
    # print("final_entities ", entities)
    nlu_result["intent"] = [
        intent
        for intent in intent_ranking
        if intent["confidence"] == max([en["confidence"] for en in intent_ranking])
    ][0]
    nlu_result["entities"] = entities
    nlu_result["intent_ranking"] = intent_ranking
    print("final   ###### :", nlu_result)
    return (nlu_result)


# path1 = "C:\\Users\\Durashi\\AppData\\Local\\Temp\\tmpp4vfq2da\\nlu"
# path2 = "C:\\Users\\Durashi\\testbot\\stack_models\\nlu-20201031-221332.tar.gz"
# path0 = "stack_models\\nlu-20201111-025247.tar.gz"
# path3 = model.get_model(path0)
# print("path3: ",path3)
# run_cmdline()
