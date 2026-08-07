import re


class PromptTemplate:
    def __init__(self, template):
        self.template = template
        self.variables = set(re.findall(r"\{(\w+)\}", template))

    def format(self, **kwargs):
        missing = self.variables - kwargs.keys()
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.template.format(**kwargs)
