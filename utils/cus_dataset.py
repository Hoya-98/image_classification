from torch.utils.data import Dataset
from PIL import Image, ImageOps

class cus_Dataset(Dataset):
    
    def __init__(self, CFG, data, transform=None):

        self.df = data
        self.transform = transform

        self.image_dir = CFG['Image_Dir']

    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        
        image = self.load_image(idx)
        label = self.load_label(idx)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

    
    def load_image(self, idx):

        img_path = f"{self.image_dir}/{self.df['file_name'][idx]}"
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        img = img.crop((self.df['x1'][idx], self.df['y1'][idx], self.df['x2'][idx], self.df['y2'][idx]))
        
        return img


    def load_label(self, idx):
        
        label = self.df['label'][idx]
        
        return label