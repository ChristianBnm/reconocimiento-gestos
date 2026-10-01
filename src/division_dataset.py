from sklearn.model_selection import train_test_split


def dividir_dataset(
    etiquetas,
    train_size,
    val_size,
    test_size,
    seed
):

    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "train_size + val_size + test_size debe ser igual a 1.0"
        )

    indices = list(range(len(etiquetas)))

    indices_train, indices_temp = train_test_split(
        indices,
        test_size=val_size + test_size,
        random_state=seed,
        stratify = etiquetas
    )

    proporcion_val = val_size / (val_size + test_size)

    etiquetas_temp = [
        etiquetas[i]
        for i in indices_temp
    ]

    indices_val, indices_test = train_test_split(
        indices_temp,
        test_size=1 - proporcion_val,
        random_state = seed,
        stratify = etiquetas_temp
    )

    return indices_train, indices_val, indices_test