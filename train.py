import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score


from model import SimpleMLP
from loss import CrossEntropyLoss
from optimizer import SGD
from utils import export_data


def train():
    X_train, y_train = export_data()
    X_test, y_test = export_data(mode="test")
    batch_size = 32
    n_epochs = 30
    kfold_train = KFold(n_splits=y_train.shape[0] // batch_size)
    kfold_test = KFold(n_splits=y_test.shape[0] // batch_size)

    model = SimpleMLP(hidden_layers=[28 * 28, 256, 64, 10])

    optimizer = SGD(model.parameters, model.grads, lr=1e-2)
    criterion = CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(f"*****************Epoch {epoch + 1}: ********************")
        train_losses = []
        train_epoch_iterator = tqdm(kfold_train.split(y_train),
                                    desc="Training (Step X) (loss=X.X)",
                                    bar_format="{l_bar}{r_bar}",
                                    dynamic_ncols=True, )
        for id, (_, index) in enumerate(train_epoch_iterator):
            X_batch_train = X_train[index].T
            y_batch_train = y_train[index]
            output = model.forward(X_batch_train)
            loss = criterion.forward(output, y_batch_train)
            train_losses.append(loss)
            train_epoch_iterator.set_description(
                "Training (Step %d / %d) (loss=%2.5f)" % (id + 1, y_train.shape[0] // batch_size, loss)
            )

            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        print("Train loss: ", sum(train_losses) / len(train_losses))
        print("Test phase: ")
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for _, index in tqdm(kfold_test.split(y_test)):
            X_batch_test = X_test[index].T
            y_batch_test = y_test[index]
            predict = np.argmax(model.forward(X_batch_test)[0], axis=0)

            accuracy_scores.append(accuracy_score(predict, y_batch_test))

        print("Test result: ")
        print(f"Accuracy: {sum(accuracy_scores) / len(accuracy_scores)}, ")


if __name__ == "__main__":
    train()
    pass
