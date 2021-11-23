import torch
import torch.nn.functional as F
import os
import json
from bin.utils import EnhancedJSONEncoder, get_best_value


def train_model(epochs, training_dataloader, validation_dataloader, model, optimizer, device, directory):
    scores = {}
    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
        validate(validation_dataloader, model, device, epoch, directory, scores)

    scores[f"Best_{get_best_value(scores)[0]}"] = get_best_value(scores)[1]
    with open(os.path.join(directory, "scores.json"), "w") as fjson:
        json.dump(scores, fjson, cls=EnhancedJSONEncoder)

def validate(validation_dataloader, model, device, epoch, directory, scores):
    best_scores = {'accuracy': 0, 'epoch': 0}
    #scores = {}
    # validate
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            model.eval()
            output = model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(validation_dataloader.dataset)
        val_acc = correct / len(validation_dataloader.dataset)

        print(
            f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
        )
        scores["Epoch_{}".format(epoch)] = val_acc
        if val_acc >= best_scores['accuracy']:
            best_scores['accuracy'] = val_acc
            #best_scores['epoch'] = epoch
            #scores["Best_Epoch_{}".format(epoch)] = val_acc
            torch.save(model.cpu().state_dict(), os.path.join(directory, "model.pt"))
