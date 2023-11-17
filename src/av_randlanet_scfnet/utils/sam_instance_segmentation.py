import os
import sys

from segment_lidar import samlidar


def run_sam_instance_segmentation(filename):
    print("Running SAM-LiDAR Instance Segmentation...")
    seg_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
    save_dir = seg_dir.replace("separate_segments", "separate_instances")
    os.makedirs(save_dir, exist_ok=True)

    files = os.listdir(seg_dir)
    for each in files:
        points = model.read(os.path.join(seg_dir, each))
        labels, *_ = model.segment(points=points)
        model.write(points=points, segment_ids=labels, save_path=os.path.join(save_dir, each))
        print("Saved instance segmentation for", each)


if __name__ == '__main__':
    run_sam_instance_segmentation(sys.argv[1])
