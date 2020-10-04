import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bndbox(img, object_points, object_titles=None):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    plt.figure("Image")

    for i, point in enumerate(object_points):
        xmin = point[0]
        ymin = point[1]
        width = point[2] - point[0]
        height = point[3] - point[1]
        rect = patches.Rectangle(xy=(xmin, ymin), width=width,
                                 height=height, linewidth=2, fill=False, edgecolor='r')
        if object_titles is not None:
            ax.text(xmin, ymin+10, object_titles[i], fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
        ax.add_patch(rect)
    plt.show()


# print(img.shape)
# print(gt_box)
# print(gt_cls)
# print(flip)
# plt.figure("Image")
# plt.imshow(transforms.ToPILImage()(img[0]))
# plt.show()
# batch_size, channels, h, w = img.shape
# for b in range(batch_size):
#     if flip[b]:
#         img[b] = torch.flip(img[b], [2])
#     plot_bndbox(transforms.ToPILImage()(img[b]), gt_box[b])
