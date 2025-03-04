# test_plot.py

import torch

from utils import attach_tiles_to_images, plot_epoch_images



@torch.no_grad()
def test_plot(model, list_tiles, device, tile_size, output_path, path_individuals=""):
    """
    Generate and plot comparison between the LR reference image, the SR texture
    and the HR texture
    """
    return
    test_lr_img, test_sr_img, test_hr_img = None, None, None

    for tile in list_tiles:
        b_lr_imgs, b_hr_texture_tile, b_s_tile = [elem.unsqueeze(0) for elem in tile]

        tile_y, tile_x = b_s_tile[0, 1:3]
        tile_y, tile_x = (tile_y.item(), tile_x.item())

        # Convert input and output using device
        b_lr_imgs = b_lr_imgs.to(device, non_blocking=True)
        b_hr_texture_tile = b_hr_texture_tile.to(device, non_blocking=True)

        test_b_sr_img_tile = model(b_lr_imgs)

        test_lr_img, test_sr_img, test_hr_img = attach_tiles_to_images(
            test_lr_img, test_sr_img, test_hr_img,
            img_size=b_s_tile[0][4:6],
            b_lr_imgs=b_lr_imgs,
            tile_size=tile_size,
            b_sr_img_tile=test_b_sr_img_tile,
            b_hr_texture=b_hr_texture_tile,
            tile_x=tile_x, tile_y=tile_y, sample_n=0
        )

    plot_epoch_images(test_lr_img, test_sr_img, test_hr_img,
                      output_path, path_individuals=path_individuals)

    return
