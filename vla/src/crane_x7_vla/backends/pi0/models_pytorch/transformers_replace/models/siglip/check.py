import transformers


def check_whether_transformers_replace_is_installed_correctly():
    # Support transformers 4.57.x (Pi0.5 adaRMS patches)
    version = transformers.__version__
    return version.startswith("4.57.")
