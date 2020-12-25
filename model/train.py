import torch
from torch import nn
import numpy as np

# Save and Load Functions


def save_checkpoint(save_path, model):
    torch.save(model, save_path)
    print(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model):
    model = torch.load(load_path)
    print(f"Model loaded from <== {load_path}")
    return model


def evaluate(model, device, iterator, valid_running_loss=0):
    model.eval()
    correct, total = 0, 0
    predictions = []
    with torch.no_grad():
        for (spelling, lexicality), _ in iterator:
            lexicality = lexicality.type(torch.LongTensor).to(device)
            spelling = spelling.type(torch.LongTensor).to(device)
            loss, logits = model(spelling, lexicality)
            valid_running_loss += loss.item()
            prediction = model.convertLogits(logits)
            correct += (lexicality - 1) == (prediction)
            total += lexicality.size
            predictions.append(prediction)
    return valid_running_loss, correct / total, np.concatenate(predictions, axis=0)


def train(
    model,
    optimizer,
    train_loader,
    valid_loader,
    test_loader,
    device,
    weight_path,
    num_epochs=3,
    eval_every=1,
    best_valid_loss=float("Inf"),
    criterion=nn.BCELoss(),
):
    test_acc_history = []
    # initialize running values
    running_loss, valid_running_loss, global_step = 0.0, 0.0, 0
    train_loss_list, valid_loss_list, global_steps_list = [], [], []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (spelling, lexicality), _ in train_loader:
            lexicality = lexicality.type(torch.LongTensor).to(device)
            spelling = spelling.type(torch.LongTensor).to(device)
            loss, _ = model(spelling, lexicality)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                valid_running_loss, accuracy, _ = evaluate(
                    model=model,
                    device=device,
                    valid_running_loss=valid_running_loss,
                    iterator=valid_loader,
                )
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                _, test_accu, _ = evaluate(
                    model=model, device=device, iterator=test_loader
                )
                test_acc_history.append(test_accu)
                running_loss, valid_running_loss = 0.0, 0.0
                model.train()

                # print progress
                print(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        global_step,
                        num_epochs * len(train_loader),
                        average_train_loss,
                        average_valid_loss,
                    )
                )

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(weight_path / "model.pt", model)

    print("Finished Training!")
    return global_steps_list, train_loss_list, valid_loss_list, test_acc_history
