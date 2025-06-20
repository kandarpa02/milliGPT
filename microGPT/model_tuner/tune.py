class microLM:
    def __init__(self, model, params):
        self.model = model
        self.loaded_params = params
        self.layer_names = list(self.loaded_params.keys())
    
    def parameters(self, freeze:list=[]):
        if freeze == []:
            return self.loaded_params
        else:
            params = {}
            for p in self.layer_names:
                if not p in freeze:
                    params[f'{p}'] = self.loaded_params[p]
            return params

    @staticmethod
    def run_fn(model, X, params:dict, num_heads):
        return model.run_fn(X, params, num_heads)
