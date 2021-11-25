import torch

from ser.transforms import transforms, normalize, flip
import numpy as np

def test_transform():
    data = np.array([[1, 2, 3], [3, 2, 8], [9, 4, 7]]).astype("float64")
    ts = [normalize]
    actual = transforms(*ts)(data)
    print(actual)
    expected = torch.Tensor([[0, 1/8, 2/8], [2/8, 1/8, 7/8], [1, 3/8, 6/8]]).float()
    expected = torch.Tensor([[1., 3., 5.], [5., 3., 15.], [17., 7., 13.]]).type(torch.float64)
    assert actual == expected


if __name__ == "__main__":
    test_transform()
    data = [[1, 2, 3], [3, 2, 8], [9, 4, 7]]
    ts = [normalize]
    actual = transforms(*ts)
    print(actual)