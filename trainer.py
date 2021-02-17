import copy
import argparse
from glob import glob
import os

import torch
from torch import nn

from PIL import Image

from model import TinyNet
from dataset import CardDataset
from transforms import train_transform, test_transform
import utils

logger = utils.get_logger('trainer')


def run_training_loop(model, optimizer, scheduler, device, train_loader, test_loader,
                      criterion_classification, criterion_localization, epochs):
    logger.info('Start training')
    for epoch in range(epochs):
        logger.debug(f'Epoch {epoch + 1}')

        model.train()
        running_loss_ce = 0.0
        running_loss_mse = 0.0
        for i, data in enumerate(train_loader):
            images, labels, bboxes_gt = data
            bboxes_gt = torch.stack(bboxes_gt, dim=1)

            images = images.to(device)
            labels = labels.to(device)
            bboxes_gt = bboxes_gt.to(device)

            optimizer.zero_grad()

            logits, bboxes = model(images)
            loss_ce = criterion_classification(logits, labels)
            loss_mse = criterion_localization(bboxes, bboxes_gt.float()) * 50
            loss = loss_ce + loss_mse
            loss.backward()
            optimizer.step()

            running_loss_ce += loss_ce.item()
            running_loss_mse += loss_mse.item()
            print_every = 5
            if (i + 1) % print_every == 0:
                running_loss_ce = running_loss_ce / print_every
                running_loss_mse = running_loss_mse / print_every
                logger.debug(f'[{epoch + 1}, {i + 1}] '
                             f'loss_ce: {running_loss_ce:.3f}, '
                             f'loss_mse {running_loss_mse:.3f}')
                running_loss_ce = 0
                running_loss_mse = 0

        scheduler.step()
        correct = 0
        total = 0
        iou = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                test_images, test_labels, test_bboxes = data
                test_images = test_images.to(device)

                outputs = model(test_images)
                _, predicted = torch.max(outputs[0].cpu().data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
                bbox_gt = [a.item() for a in test_bboxes]

                bbox = outputs[1].cpu().data.numpy().flatten()
                iou += utils.get_iou(bbox_gt, bbox)

        iou = iou / total
        accuracy = 100 * correct / total
        logger.info(f'Test -- Accuracy: {accuracy:.4f}, IoU: {iou:.4f}')


def start_training(train_image_folder: str, train_csv_path: str, test_image_folder: str, test_csv_path: str,
                   model_path: str, epochs=50, use_gpu=False) -> None:
    model = TinyNet()

    train_dataset = CardDataset(
        image_folder=train_image_folder,
        csv_path=train_csv_path,
        transform=train_transform,
    )

    test_dataset = CardDataset(
        image_folder=test_image_folder,
        csv_path=test_csv_path,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=7
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=7
    )

    criterion_classification = nn.CrossEntropyLoss()
    criterion_localization = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion_classification=criterion_classification,
        criterion_localization=criterion_localization,
        epochs=epochs,
    )

    model_half = copy.deepcopy(model).half()
    torch.save(model_half.state_dict(), model_path)
    logger.info(f'Model has successfully saved: {model_path}')


def run_inference(model_half_path, test_image_folder, result_folder):
    model = TinyNet()
    model.load_state_dict(torch.load(model_half_path))
    model = model.float()

    test_folder_pattern = os.path.join(test_image_folder, '*')
    file_paths = sorted(glob(test_folder_pattern))

    for path in file_paths:
        img = Image.open(path).convert('RGBA').convert('RGB')
        filename = path.split('/')[-1]
        logger.debug(f'Processing: {filename}')

        x = test_transform(img)
        x = x.unsqueeze(0)
        logits, bbox_pred = model(x)
        label = logits.cpu().detach()[0].numpy().argmax()
        text = 'visa' if label == 1 else 'mastercard'

        bbox_pred = bbox_pred.cpu().detach()[0].numpy()
        img_bbox = utils.get_image_with_bbox_and_text(img, bbox_pred, text)

        result_path = os.path.join(result_folder, filename)
        img_bbox.save(result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", type=bool, default=True)
    parser.add_argument("--train-image-folder", type=str, default='../data/train')
    parser.add_argument("--train-csv-path", type=str, default='../data/train.csv')
    parser.add_argument("--test-image-folder", type=str, default='../data/test')
    parser.add_argument("--test-csv-path", type=str, default='../data/test.csv')
    parser.add_argument("--result-folder", type=str, default='../result')
    parser.add_argument("--model-path", type=str, default='../model_half.pt')
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    if not args.skip_training:
        start_training(
            train_image_folder=args.train_image_folder,
            train_csv_path=args.train_csv_path,
            test_image_folder=args.test_image_folder,
            test_csv_path=args.test_csv_path,
            model_path=args.model_path,
            epochs=args.epochs,
        )
    else:
        logger.info(f'Skip training. Model: {args.model_path}')

    run_inference(
        model_half_path=args.model_path,
        test_image_folder=args.test_image_folder,
        result_folder=args.result_folder
    )
