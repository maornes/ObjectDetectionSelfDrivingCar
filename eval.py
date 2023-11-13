from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter
from datasets import dataset, collate_fn

pp = PrettyPrinter()

test_dataset = dataset(csv_file = '/home/matino/anaconda3/envs/testenv/adat_kaggle/labels_val.csv',
                           root_dir = '/home/matino/anaconda3/envs/testenv/adat_kaggle/images/',
                           transform = transform)

test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn=collate_fn)

# Parameters
batch_size = 16
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_ssd300.pth.tar'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    #det_boxes_300 = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    #true_boxes_300 = list()
    true_boxes = list()
    true_labels = list()
    
    k = 0

    with torch.no_grad():
        # Batches
        #for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
        for i, (images, boxes, labels) in enumerate(tqdm(train_loader, desc='Evaluating')):
            if k == 0:
                images = images.to(device)  # (N, 3, 300, 300)


                # Forward prop.
                predicted_locs, predicted_scores = model(images)

                # Detect objects in SSD output
                det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                           min_score=0.01, max_overlap=0.45,
                                                                                           top_k=200)
                
                #det_boxes_batch_300 = []
                #boxes_300 = []

                #for l in det_boxes_batch:
                    #det_boxes_batch_l = l * 300
                    #det_boxes_batch_300.append(det_boxes_batch_l)
                    
                #for h in boxes:
                    #boxes_h = h * 300
                    #boxes_300.append(boxes_h)
                
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                #det_boxes_300.extend(det_boxes_batch_300)
                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                #true_boxes_300.extend(boxes_300)
                true_boxes.extend(boxes)
                true_labels.extend(labels)

                k = k + 1
    
    # Calculate evaluation metrics
    confusion_matrices, total_confusion_matrix, precisions_recalls, total_precision, total_recall, average_precisions, mean_average_precision, F1, total_F1 = evaluation_metrics(det_boxes, det_labels, det_scores, true_boxes, true_labels)

    # Print the evaluation metrics
    print('Confusion matrices at the level of classes')
    pp.pprint(confusion_matrices)
    
    print('\nTotal confusion matrix:', total_confusion_matrix)
    
    print('\nPrecisions and recalls at the level of classes')
    pp.pprint(precisions_recalls)
    
    print('\nTotal precision:', total_precision)
    
    print('\nTotal recall:', total_recall)
    
    print('\nAverage precisions at the level of classes')
    pp.pprint(average_precisions)
    
    print('\nMean Average Precision (mAP):')
    pp.pprint(mean_average_precision)
    
    print('\nF1 at the level of classes:')
    pp.pprint(F1)
    
    print('\nTotal F1:', total_F1)
    
    return images, det_boxes, det_labels, det_scores, confusion_matrices, total_confusion_matrix, precisions_recalls, total_precision, total_recall, average_precisions, mean_average_precision, F1, total_F1

if __name__ == '__main__':
    #images, det_boxes, det_labels, det_scores, confusion_matrices, precisions_recalls, total_precision, total_recall, average_precisions, mean_average_precision, total_F1 = evaluate(test_loader, model)
    images, det_boxes, det_labels, det_scores, confusion_matrices, total_confusion_matrix, precisions_recalls, total_precision, total_recall, average_precisions, mean_average_precision, F1, total_F1 = evaluate(train_loader, model)
