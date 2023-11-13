import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms

# Make neccesary changes for the SSD model
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([transforms.Resize((300, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

class dataset(Dataset):
    # Initialize the dataset
    def __init__(self, csv_file, root_dir, transform):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.unique_image_paths = self.data['frame'].unique()
        #self.unique_image_paths = self.data['frame'].unique()[:2000] # first 100 images \n",
        self.transforms = transform
        self.image_width = 480
        self.image_height = 300

    # Get item method to retrieve data for a given index
    def __getitem__(self, index):
        image_filename = self.unique_image_paths[index]
        image_path = os.path.join(self.root_dir, image_filename)
        image_data = self.data[self.data['frame'] == image_filename]

        image = Image.open(image_path).convert('RGB')
        boxes, labels = self.load_annotations(image_data)

        image = self.transforms(image)
        
        return image, boxes, labels

    # Examine the number of images
    def __len__(self):
        return len(self.unique_image_paths)

    # Format bounding box coordinates and labels
    def load_annotations(self, image_data):
        xmin = image_data['xmin'].tolist()
        xmax = image_data['xmax'].tolist()
        ymin = image_data['ymin'].tolist()
        ymax = image_data['ymax'].tolist()
        class_ids = image_data['class_id'].tolist()

        boxes = []
        labels = []
        for x1, x2, y1, y2, cid in zip(xmin, xmax, ymin, ymax, class_ids):
            normalized_box = [
                x1 / self.image_width, 
                y1 / self.image_height,
                x2 / self.image_width, 
                y2 / self.image_height 
            ]
            boxes.append(normalized_box)
            labels.append(cid)
            
        return torch.FloatTensor(boxes), torch.LongTensor(labels)
    
        # Examine the presence of missing bounding boxes
    def find_missing_values_bounding_boxes(self):
        missing_values = 0

        for index in range(len(self)):
            _, boxes, _ = self[index]

            if len(boxes) == 0:
                missing_values += 1

        return missing_values

    # Examine the presence of missing labels
    def find_missing_values_labels(self):
        missing_values = 0

        for index in range(len(self)):
            _, _, labels = self[index]

            if len(labels) == 0:
                missing_values += 1

        return missing_values

    # Descriptive statistics of image sizes
    def summarize_image_sizes(self):
        image_sizes = [image.size() for image, _, _ in self]

        width_values = [size[2] for size in image_sizes]
        height_values = [size[1] for size in image_sizes]

        size_summary = {
            'min_width': min(width_values),
            'max_width': max(width_values),
            'mean_width': sum(width_values) / len(width_values),
            'min_height': min(height_values),
            'max_height': max(height_values),
            'mean_height': sum(height_values) / len(height_values),
        }

        return size_summary

    # Examine the number of labels belonging to an image
    def plot_label_per_image_distribution(self):
        label_counts = []
        for index in range(len(self)):
            _, _, labels = self[index]
            labels_per_image = len(labels)
            label_counts.append(labels_per_image)

        plt.hist(label_counts, bins=max(label_counts) - min(label_counts) + 1)
        plt.xlabel("Number of labels")
        plt.ylabel("Frequency")
        plt.title("Distribution of label counts per image")
        plt.show()

    # Examine the number of labels in each class
    def plot_label_distribution(self):
        label_names = {
            1: 'car',
            2: 'truck',
            3: 'pedestrian',
            4: 'bicyclist',
            5: 'light'
        }
        
        label_counts = self.data['class_id'].map(label_names).value_counts()
        labels = label_counts.index
        counts = label_counts.values

        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Distribution of labels")

        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom')

        plt.show()

    # Examine the distribution of sizes of bounding boxes
    def plot_bounding_box_size_distribution(self):
        box_sizes = []

        for index in range(len(self)):
            _, boxes, _ = self[index]

            for box in boxes:
                box_width = (box[2] - box[0]) * self.image_width
                box_height = (box[3] - box[1]) * self.image_height
                box_size = box_width * box_height
                box_sizes.append(box_size)

        plt.hist(box_sizes, bins=20)
        plt.xlabel("Bounding box size")
        plt.ylabel("Frequency")
        plt.title("Distribution of bounding box sizes in pixels")
        plt.show()

def collate_fn(batch):

    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)

    return images, boxes, labels
