import torch
from ser.utils import generate_ascii_art

def perform_infer(model, images):
    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))

    return (pred, certainty)