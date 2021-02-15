import copy
import logging
import argparse

import torch
from torch import nn
import torchvision.transforms as transforms

from model import TinyNet
from dataset import CardDataset
import utils

logger = utils.get_logger('trainer')


def run_training_loop(model, optimizer, scheduler, device, train_loader, test_loader,
                      criterion_classification, criterion_localization, epochs):
    logger.info('Start training')
    for epoch in range(epochs):
        logger.debug(f'Epoch {epoch}')

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
                   model_path: str, epochs=50, image_size=(112, 184), use_gpu=False) -> None:
    model = TinyNet()

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=1, hue=0.1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

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
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    criterion_classification = nn.CrossEntropyLoss()
    criterion_localization = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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


if __name__ == '__main__':
    parser = utils.get_args_parser()
    args = parser.parse_args()

    start_training(
        train_image_folder=args.train_image_folder,
        train_csv_path=args.train_csv_path,
        test_image_folder=args.test_image_folder,
        test_csv_path=args.test_csv_path,
        model_path=args.model_path,
        epochs=args.epochs,
    )
