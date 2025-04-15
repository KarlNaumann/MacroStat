import argparse

from macrostat.models import get_available_models, get_model_classes

MODELGROUPS = ["GL06"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docdir", type=str, default="docs")
    args = parser.parse_args()

    for modelname in get_available_models():
        print(f"Pre-processing model {modelname}")

        group = [i for i in MODELGROUPS if i in modelname]
        subdir = modelname if len(group) == 0 else f"{group[0]}/{modelname}"
        modeldir = f"{args.docdir}/models/{subdir}"

        print("  - Creating Parameter table")
        model_classes = get_model_classes(modelname=modelname)
        parameters = model_classes.Parameters()
        df = parameters.to_csv(f"{modeldir}/parameters.csv", sphinx_math=True)

        # Intermediate step: we separate parameters and hyperparameters
        params = df[df["ParameterType"] == "Parameter"]
        params = params.drop(columns=["ParameterType"])
        params.to_csv(f"{modeldir}/parameters.csv")
        hyper = df[df["ParameterType"] == "HyperParameter"]
        hyper = hyper.drop(columns=["ParameterType", "Lower Bound", "Upper Bound"])
        hyper.to_csv(f"{modeldir}/hyperparameters.csv")

        # Create the parameters.rst file
        version_str = f"Parameters {modelname}"
        rst = [
            len(version_str) * "=",
            version_str,
            len(version_str) * "=",
            "\n",
            f"The following two tables contain the default values for the hyperparameters and parameters of the {modelname} model.",
            "\n",
            "Hyperparameters",
            len("Hyperparameters") * "=",
            ".. csv-table::",
            "\t:file: hyperparameters.csv",
            "\t:header-rows: 1",
            "\n",
            "Parameters",
            len("Parameters") * "=",
            ".. csv-table::",
            "\t:file: parameters.csv",
            "\t:header-rows: 1",
        ]

        with open(f"{modeldir}/parameters.rst", "w") as f:
            f.write("\n".join(rst) + "\n")

        print("  - Creating Variables table")
        variables = model_classes.Variables()
        variables.info_to_csv(f"{modeldir}/variables.csv", sphinx_math=True)

        version_str = f"Variables {modelname}"
        rst = [
            len(version_str) * "=",
            version_str,
            len(version_str) * "=",
            "\n",
            f"The following table contains the default information for the variables of the {modelname} model.",
            "\n",
            ".. csv-table::",
            "\t:file: variables.csv",
            "\t:header-rows: 1",
        ]

        with open(f"{modeldir}/variables.rst", "w") as f:
            f.write("\n".join(rst) + "\n")

        print("  - Creating Balance Sheet table")
        balance_sheet = variables.balance_sheet_theoretical(
            mathfmt="myst", non_camel_case=True
        )
        balance_sheet.to_csv(f"{modeldir}/balance_sheet.csv")

        print("  - Creating Transaction Matrix table")
        transaction_matrix = variables.transaction_matrix_theoretical(
            mathfmt="myst", non_camel_case=True
        )
        transaction_matrix.to_csv(f"{modeldir}/transaction_matrix.csv")
