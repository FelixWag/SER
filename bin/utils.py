import dataclasses, json

"""
Utitlity functions
"""

# Code from Stackoverflow: https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

def get_best_value(scores):
    """

    :param scores:
    :return: Highest values in a dictionary and corresponing key
    """
    best_value = scores[max(scores, key=scores.get)]
    best_epoch = list(scores.keys())[list(scores.values()).index(best_value)][0]

    return (best_epoch, best_value)
