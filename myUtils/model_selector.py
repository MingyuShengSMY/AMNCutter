from models.AMNCutter import AMNCutter


def model_selector(config):
    if config.method_name == "AMNCutter":
        model = AMNCutter(config)
    else:
        raise ValueError(f"Unknown method name: {config.method_name}")

    return model


