import os
import sys

samlidar_pythonpath = "/home/pc1/miniconda3/envs/samlidar/bin/python"
sys.path.append(samlidar_pythonpath)
os.environ["PYTHONPATH"] = samlidar_pythonpath

try:
    sys.path.index(samlidar_pythonpath)    # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(samlidar_pythonpath)    # Or os.getcwd() for this directory


def run_sam_instance_segmentation(filename):
    from segment_lidar import samlidar
    # import samlidar
    print("Running SAM-LiDAR Instance Segmentation for", filename)
    seg_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
    save_dir = seg_dir.replace("separate_segments", "separate_instances")
    os.makedirs(save_dir, exist_ok=True)

    files = os.listdir(seg_dir)
    for each in files:
        try:
            points = model.read(os.path.join(seg_dir, each))
            labels, *_ = model.segment(points=points)
            model.write(points=points, segment_ids=labels, save_path=os.path.join(save_dir, each))
            print("Saved instance segmentation for", each)
        except Exception as err:
            print(err)


if __name__ == '__main__':
    run_sam_instance_segmentation(sys.argv[1])
