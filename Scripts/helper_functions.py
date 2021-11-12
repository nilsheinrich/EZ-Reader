def load_parameters_pandas(file="none"):
    import pandas as pd
    partable = pd.read_csv(
        file, names=["parameter", "value"], header=None, sep="\s+")
    # the following parameters must be included and set in partable
    known_pars = pd.Series(["A", "Alpha1", "Alpha2", "Alpha3", "Delta", "Epsilon", "Eta1", "Eta2", "Lambda", "M1", "M2", "Omega1",
                            "Omega2", "I", "ITarget", "pF", "pFTarget", "Psi", "S", "SigmaGamma", "V", "Xi", "maxLength", "maxSentenceLength", "NSubjects"])
    if not all(known_pars.isin(partable.parameter)):
        raise CustomError("Something wrong with parameter file")
        # also check if multiple lines contain the same parameter name
    else:
        return partable


def load_parameters_dict(file="none", parameters={}):
    with open(file) as fh:
        for line in fh:
            name, value = line.strip().split()
            parameters[name] = float(value)
    # the following parameters mustbe included and set in partable
    known_pars = {"A", "Alpha1", "Alpha2", "Alpha3", "Delta", "Epsilon", "Eta1", "Eta2", "Lambda", "M1", "M2", "Omega1", "Omega2",
                  "I", "ITarget", "pF", "pFTarget", "Psi", "S", "SigmaGamma", "V", "Xi", "maxLength", "maxSentenceLength", "NSubjects"}
    if known_pars <= parameters.keys():
        return parameters
    else:
        raise CustomError("Something wrong with parameters in parameter file")


# def update_parameters(parameters, names_of_updated, new_values):

#     for index,row in new_parameters.iterrow():
#         all_parameters
#     return updated_parameters

# def rescale_parameters(parname, parval):


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    # shamelessly copied from " https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory/31809973#31809973 "

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise
