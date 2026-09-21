import copy


class VerificationError(Exception):
    """Raised when a tool reported success and a check of the world says otherwise.

    Deliberately not the same thing as ToolError. A tool that raises ToolError ran, decided it
    could not do the job, and nothing happened. A tool that fails verification said it had done
    the job -- so something may well have happened, just not the thing that was reported.
    """


def verified(tool, check):
    """A copy of `tool` that looks at the world after it runs, instead of taking its word.

    `check(args, value)` gets the arguments the tool was called with and whatever it returned.
    It returns None when all is well, or a message saying what is actually true -- and that
    message goes to the model, so it is worth writing as a fact rather than as a complaint.

    A copy, not a change to the original: the tool being wrapped may be registered elsewhere, or
    handed to another agent that verifies it differently, or not at all.
    """
    def call(**kwargs):
        value = tool.function(**kwargs)
        problem = check(kwargs, value)
        if problem:
            raise VerificationError(problem)
        return value

    checked = copy.copy(tool)
    checked.function = call
    # The schema, the argument check and the name all come off `args` and `name`, which the copy
    # keeps -- so what the model is shown is unchanged. Verification is not part of the deal the
    # tool offers; it is something the caller does about the answer.
    return checked
