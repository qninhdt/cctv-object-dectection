from torchvision.transforms import functional as F


class SquarePad(object):
    def __call__(self, sample):
        image = sample["image"]
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        image = F.pad(image, padding, 0, "constant")

        return {**sample, "image": image}
