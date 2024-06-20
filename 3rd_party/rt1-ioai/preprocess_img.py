from PIL import Image
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


class PreprocessImg(Dataset):
    def __init__(self, data_path, cams, types, img_size=(320, 256)):
        self.data_path = data_path
        self.transform = Resize(img_size[::-1])

        self.src_list = []
        self.dst_list = []
        for ep in os.listdir(data_path):
            if not ep.startswith('episode_'):
                continue
            for cam in cams:
                for t in types:
                    src_path = os.path.join(data_path, ep, cam, t)
                    dst_path = src_path + '_processed'
                    os.makedirs(dst_path, exist_ok=True)
                    for img_name in os.listdir(src_path):
                        self.src_list.append(os.path.join(src_path, img_name))
                        self.dst_list.append(os.path.join(dst_path, img_name))

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        src_path = self.src_list[idx]
        dst_path = self.dst_list[idx]
        img = Image.open(src_path)
        img = self.transform(img)
        img.save(dst_path)
        return None


if __name__ == '__main__':
    data_path = '/home/caohaiheng/tmp'
    cams = ['base_camera', 'gripper_camera']
    types = ['rgb']
    dataset = PreprocessImg(data_path, cams, types)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=20, collate_fn=lambda x: x)
    for data in tqdm(dataloader):
        pass
