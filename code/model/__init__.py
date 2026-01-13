from .agent import DeepSpeedAgent
from .openllama import OpenLLAMAPEFTModel
from .openllama import MCPandaModel

def load_model(args):
    agent_name = args['models'][args['model']]['agent_name']
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    agent = globals()[agent_name](model, args)
    return agent
