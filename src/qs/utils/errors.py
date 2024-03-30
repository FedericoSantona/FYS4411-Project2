class NotInitialized(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """

    pass


class NotTrained(Exception):
    r"""Failed attempt at accessing a trained model.
    The model needs to be trained first.
    """

    pass


class SamplingNotPerformed(Exception):
    r"""Failed attempt at accessing posterior samples.
    A call to the sample method must be carried out first.
    """

    pass
