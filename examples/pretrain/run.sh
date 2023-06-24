python segmentation/train_segmentation.py --use_img --cat laptop --run half --arch pn --half
python segmentation/train_segmentation.py --use_img --cat laptop --run full --arch pn
python segmentation/train_segmentation.py --use_img --cat laptop --run full --arch mpn
python segmentation/train_segmentation.py --use_img --cat laptop --run full --arch lpn
python reconstruction/train_reconstruction.py --use_img --cat laptop --run full
python simsiam/train_simsiam.py --use_img --cat laptop --run full
# python train_segmentation.py --use_img --cat laptop --run 0 --arch pn --vis pn_0.pth  # visualize the result