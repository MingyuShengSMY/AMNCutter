class Dict2Class:
    def __init__(self, input_dict: dict):
        for k in input_dict:
            v = input_dict[k]
            if isinstance(v, dict):
                v = Dict2Class(v)
            if "comment" in k:
                continue
            setattr(self, k, v)
