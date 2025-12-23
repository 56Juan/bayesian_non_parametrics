from sklearn.model_selection import train_test_split

def split_data(
    X,
    y,
    test_size=0.2,
    val_size=None,
    random_state=123
):
    """
    Divide los datos en train / validation / test.

    Si val_size es None:
        -> train / test
    Si val_size se especifica:
        -> train / validation / test
    """

    # Primero: train + val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    if val_size is None:
        return {
            "X_train": X_train_val,
            "y_train": y_train_val,
            "X_test": X_test,
            "y_test": y_test
        }

    # Ajustar proporción de validation respecto a train+val
    val_ratio = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }
